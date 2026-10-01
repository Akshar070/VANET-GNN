import os
import yaml
import time
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.kmeans_baseline import run_kmeans, scale_features
from src.evaluation.clustering_metrics import calculate_clustering_metrics, calculate_intra_cluster_distance
from src.evaluation.topology_metrics import calculate_topology_metrics
from src.evaluation.temporal_metrics import calculate_temporal_overlap

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_kmeans_pipeline():
    print("--- PHASE 4: K-MEANS BASELINE CLUSTERING ---")
    start_time = time.time()
    
    config = load_config()
    kmeans_cfg = config['clustering']['kmeans']
    scale_cfg = config['clustering']['scaling']
    ksel_cfg = config['clustering']['k_selection']
    
    graphs_dir = Path("data/processed/graphs")
    meta_path = graphs_dir / "graph_metadata.csv"
    
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run Phase 3 first.")
        return
        
    meta_df = pd.read_csv(meta_path)
    windows = meta_df['window_id'].sort_values().tolist()
    
    if not windows:
        print("Error: No graphs found.")
        return
        
    print(f"Loaded {len(windows)} graph snapshots.")
    
    # Representative windows for K-selection and visualization
    rep_indices = [
        0,
        len(windows) // 4,
        len(windows) // 2,
        3 * len(windows) // 4,
        len(windows) - 1
    ]
    rep_windows = sorted(list(set([windows[i] for i in rep_indices if i < len(windows)])))
    
    # ---------------------------------------------------------
    # K-SELECTION
    # ---------------------------------------------------------
    k_sel_results = []
    if ksel_cfg['enabled']:
        print("Running K-selection diagnostics...")
        for wid in rep_windows:
            graph_path = graphs_dir / f"graph_{wid:04d}.gpickle"
            if not graph_path.exists(): continue
            
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
                
            if G.number_of_nodes() < 2:
                continue
                
            coords = []
            for n, d in G.nodes(data=True):
                coords.append([d['x'], d['y'], d['speed']])
            X = np.array(coords)
            X_scaled = scale_features(X, method=scale_cfg['method'])
            
            max_possible_k = min(ksel_cfg['max_k'], len(X) - 1)
            for test_k in range(max(2, ksel_cfg['min_k']), max_possible_k + 1):
                labels, _ = run_kmeans(X_scaled, k=test_k, random_state=kmeans_cfg['random_state'],
                                       n_init=kmeans_cfg['n_init'], max_iter=kmeans_cfg['max_iter'])
                metrics = calculate_clustering_metrics(X_scaled, labels)
                
                k_sel_results.append({
                    'window_id': wid,
                    'k': test_k,
                    'silhouette_score': metrics['silhouette_score'],
                    'davies_bouldin_score': metrics['davies_bouldin_score'],
                    'calinski_harabasz_score': metrics['calinski_harabasz_score']
                })
                
        if k_sel_results:
            Path("results/tables").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(k_sel_results).to_csv("results/tables/kmeans_k_selection.csv", index=False)
            print("Saved K-selection diagnostics.")
            
    # ---------------------------------------------------------
    # MAIN CLUSTERING
    # ---------------------------------------------------------
    print(f"Running K-Means (K={kmeans_cfg['k']}) for all graphs...")
    
    out_clustering_dir = Path("data/processed/clustering/kmeans")
    out_clustering_dir.mkdir(parents=True, exist_ok=True)
    
    out_tables_dir = Path("results/tables")
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    
    assignments = []
    cluster_stats = []
    metrics_list = []
    topology_list = []
    inertia_list = []
    
    vids_history = {} # For temporal overlap
    
    for wid in windows:
        graph_path = graphs_dir / f"graph_{wid:04d}.gpickle"
        if not graph_path.exists(): continue
        
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
            
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            vids_history[wid] = []
            continue
            
        vids = []
        coords = []
        for n, d in G.nodes(data=True):
            vids.append(n)
            coords.append([d['x'], d['y'], d['speed']])
            
        X = np.array(coords)
        vids_history[wid] = vids
        
        if n_nodes < kmeans_cfg['k']:
            print(f"Warning: Window {wid} has {n_nodes} nodes, less than K={kmeans_cfg['k']}")
            
        # Feature scaling
        X_scaled = scale_features(X, method=scale_cfg['method'])
        
        # Clustering
        labels, inertia = run_kmeans(X_scaled, k=kmeans_cfg['k'], 
                                     random_state=kmeans_cfg['random_state'],
                                     n_init=kmeans_cfg['n_init'], 
                                     max_iter=kmeans_cfg['max_iter'])
                                     
        # Save Assignments
        for i, vid in enumerate(vids):
            assignments.append({
                'window_id': wid,
                'vehicle_id': vid,
                'x': X[i, 0],
                'y': X[i, 1],
                'speed': X[i, 2],
                'cluster': labels[i]
            })
            
        # Cluster Statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_sz = counts.min()
        max_sz = counts.max()
        mean_sz = counts.mean()
        std_sz = counts.std() if len(counts) > 1 else 0.0
        
        for lbl, count in zip(unique_labels, counts):
            idx = (labels == lbl)
            c_x = X[idx, 0].mean()
            c_y = X[idx, 1].mean()
            c_speed = X[idx, 2].mean()
            
            cluster_stats.append({
                'window_id': wid,
                'cluster': lbl,
                'number_of_clusters': len(unique_labels),
                'vehicle_count': count,
                'min_cluster_size': min_sz,
                'max_cluster_size': max_sz,
                'mean_cluster_size': mean_sz,
                'std_cluster_size': std_sz,
                'mean_x': c_x,
                'mean_y': c_y,
                'mean_speed': c_speed
            })
            
        # Metrics
        c_metrics = calculate_clustering_metrics(X_scaled, labels)
        c_metrics['window_id'] = wid
        c_metrics['k'] = kmeans_cfg['k']
        c_metrics['intra_cluster_distance'] = calculate_intra_cluster_distance(X, labels)
        metrics_list.append(c_metrics)
        
        # Topology
        t_metrics = calculate_topology_metrics(G, labels, vids)
        t_metrics['window_id'] = wid
        t_metrics['k'] = kmeans_cfg['k']
        topology_list.append(t_metrics)
        
        # Inertia
        inertia_list.append({
            'window_id': wid,
            'k': kmeans_cfg['k'],
            'inertia': inertia
        })
        
    # Save Assignments
    pd.DataFrame(assignments).to_csv(out_clustering_dir / "kmeans_assignments.csv", index=False)
    
    # Save Tables
    pd.DataFrame(cluster_stats).to_csv(out_tables_dir / "kmeans_cluster_statistics.csv", index=False)
    
    metrics_df = pd.DataFrame(metrics_list)
    # Reorder columns slightly
    m_cols = ['window_id', 'k', 'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score', 'intra_cluster_distance']
    metrics_df = metrics_df[[c for c in m_cols if c in metrics_df.columns]]
    metrics_df.to_csv(out_tables_dir / "kmeans_metrics.csv", index=False)
    
    top_df = pd.DataFrame(topology_list)
    top_df.to_csv(out_tables_dir / "kmeans_topology_metrics.csv", index=False)
    
    pd.DataFrame(inertia_list).to_csv(out_tables_dir / "kmeans_inertia.csv", index=False)
    
    # Temporal Overlap
    temporal_overlaps = []
    for i in range(len(windows) - 1):
        wid_curr = windows[i]
        wid_next = windows[i+1]
        overlap = calculate_temporal_overlap(vids_history[wid_curr], vids_history[wid_next])
        overlap['window_id'] = wid_curr
        overlap['next_window_id'] = wid_next
        temporal_overlaps.append(overlap)
        
    pd.DataFrame(temporal_overlaps).to_csv(out_tables_dir / "kmeans_temporal_overlap.csv", index=False)
    
    # ---------------------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------------------
    print("Generating visualizations...")
    out_fig_dir = Path("results/figures/kmeans")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    
    assign_df = pd.DataFrame(assignments)
    
    # 1. Representative Windows
    for wid in rep_windows:
        w_df = assign_df[assign_df['window_id'] == wid]
        if w_df.empty: continue
        
        plt.figure(figsize=(10, 8))
        
        # Optionally overlay edges if we want the diagnostic plot
        graph_path = graphs_dir / f"graph_{wid:04d}.gpickle"
        if graph_path.exists():
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            pos = {n: (d['x'], d['y']) for n, d in G.nodes(data=True)}
            nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='gray')
            
        # Scatter points colored by cluster
        scatter = plt.scatter(w_df['x'], w_df['y'], c=w_df['cluster'], cmap='tab10', s=30, alpha=0.8)
        plt.title(f"K-Means Clusters (Window {wid}, K={kmeans_cfg['k']})")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_fig_dir / f"kmeans_window_{wid:04d}.png")
        plt.close()
        
    # 2. Cluster Size Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(assign_df.groupby(['window_id', 'cluster']).size(), bins=30, edgecolor='black', alpha=0.7)
    plt.title('Cluster Size Distribution (All Windows)')
    plt.xlabel('Cluster Size (Number of Vehicles)')
    plt.ylabel('Frequency')
    plt.savefig(out_fig_dir / "cluster_size_distribution.png")
    plt.close()
    
    # 3. Cluster count over time (though it's fixed K here, good for diagnostic)
    c_stats_df = pd.DataFrame(cluster_stats)
    if not c_stats_df.empty:
        w_counts = c_stats_df.groupby('window_id')['cluster'].nunique()
        plt.figure(figsize=(10, 5))
        plt.plot(w_counts.index, w_counts.values, marker='o')
        plt.title('Number of Clusters over Time')
        plt.xlabel('Window ID')
        plt.ylabel('Cluster Count')
        plt.ylim(0, kmeans_cfg['k'] + 1)
        plt.savefig(out_fig_dir / "cluster_count_over_time.png")
        plt.close()
        
    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    summary = {
        'number_of_windows': len(windows),
        'k': kmeans_cfg['k'],
        'average_silhouette': metrics_df['silhouette_score'].mean(),
        'average_davies_bouldin': metrics_df['davies_bouldin_score'].mean(),
        'average_calinski_harabasz': metrics_df['calinski_harabasz_score'].mean(),
        'average_intra_cluster_distance': metrics_df['intra_cluster_distance'].mean(),
        'average_internal_edge_ratio': top_df['internal_edge_ratio'].mean(),
        'average_cluster_size': c_stats_df['mean_cluster_size'].mean(),
        'average_cluster_size_std': c_stats_df['std_cluster_size'].mean()
    }
    pd.DataFrame([summary]).to_csv(out_tables_dir / "kmeans_summary.csv", index=False)
    
    end_time = time.time()
    
    print("\n==========================================")
    print("PHASE 4 SUMMARY")
    print("==========================================")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Average time per graph: {(end_time - start_time) / len(windows):.4f} seconds")
    print(f"Number of graphs processed: {len(windows)}")
    print(f"K value used: {kmeans_cfg['k']}")
    print(f"Feature vector: [x, y, speed]")
    print(f"Scaling method: {scale_cfg['method']}")
    print(f"Random state: {kmeans_cfg['random_state']}, n_init: {kmeans_cfg['n_init']}")
    print(f"\nAverage silhouette: {summary['average_silhouette']:.4f}")
    print(f"Average Davies-Bouldin: {summary['average_davies_bouldin']:.4f}")
    print(f"Average Calinski-Harabasz: {summary['average_calinski_harabasz']:.4f}")
    print(f"Average intra-cluster dist: {summary['average_intra_cluster_distance']:.2f} m")
    print(f"Average internal edge ratio: {summary['average_internal_edge_ratio']:.4f}")
    print(f"Average cluster size: {summary['average_cluster_size']:.2f}")
    print("==========================================")
    print("\nPhase 4 complete.")

if __name__ == "__main__":
    run_kmeans_pipeline()
