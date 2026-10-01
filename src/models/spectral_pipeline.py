import os
import yaml
import time
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.spectral_baseline import (
    run_spectral_clustering, 
    calculate_eigenvalues, 
    construct_adjacency_from_graph
)
from src.models.kmeans_baseline import scale_features
from src.evaluation.clustering_metrics import calculate_clustering_metrics
from src.evaluation.topology_metrics import calculate_topology_metrics, calculate_modularity, calculate_conductance
from src.evaluation.temporal_metrics import calculate_temporal_overlap

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_spectral_pipeline():
    print("--- PHASE 5: SPECTRAL CLUSTERING BASELINE ---")
    start_time = time.time()
    
    config = load_config()
    spectral_cfg = config['clustering']['spectral']
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
    
    rep_indices = [
        0,
        len(windows) // 4,
        len(windows) // 2,
        3 * len(windows) // 4,
        len(windows) - 1
    ]
    rep_windows = sorted(list(set([windows[i] for i in rep_indices if i < len(windows)])))
    
    out_clustering_dir = Path("data/processed/clustering/spectral")
    out_clustering_dir.mkdir(parents=True, exist_ok=True)
    
    out_tables_dir = Path("results/tables")
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    
    out_fig_dir = Path("results/figures/spectral")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # K-SELECTION & EIGENVALUES
    # ---------------------------------------------------------
    k_sel_results = []
    eig_results = []
    eigengap_results = []
    
    if ksel_cfg['enabled']:
        print("Running Spectral K-selection and Eigenvalue analysis...")
        for wid in rep_windows:
            graph_path = graphs_dir / f"graph_{wid:04d}.gpickle"
            if not graph_path.exists(): continue
            
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
                
            n_nodes = G.number_of_nodes()
            if n_nodes < 2:
                continue
                
            vids = list(G.nodes())
            A = construct_adjacency_from_graph(G, vids)
            
            # Eigenvalues
            eigenvals = calculate_eigenvalues(A)
            for i, val in enumerate(eigenvals[:20]): # Save up to top 20
                eig_results.append({
                    'window_id': wid,
                    'eigenvalue_index': i + 1,
                    'eigenvalue': val
                })
                
            for k in range(1, len(eigenvals) - 1):
                if k <= ksel_cfg['max_k']:
                    gap = eigenvals[k] - eigenvals[k-1]
                    eigengap_results.append({
                        'window_id': wid,
                        'k': k,
                        'gap': gap
                    })
            
            # K-Selection
            coords = []
            for n in vids:
                d = G.nodes[n]
                coords.append([d['x'], d['y'], d['speed']])
            X = np.array(coords)
            X_scaled = scale_features(X, method=scale_cfg['method'])
            
            max_possible_k = min(ksel_cfg['max_k'], n_nodes - 1)
            for test_k in range(max(2, ksel_cfg['min_k']), max_possible_k + 1):
                labels = run_spectral_clustering(A, n_clusters=test_k, 
                                                 random_state=spectral_cfg['random_state'],
                                                 n_init=spectral_cfg['n_init'], 
                                                 assign_labels=spectral_cfg['assign_labels'])
                                                 
                metrics = calculate_clustering_metrics(X_scaled, labels)
                mod = calculate_modularity(G, labels, vids)
                
                # Internal edge ratio
                top_metrics = calculate_topology_metrics(G, labels, vids)
                
                k_sel_results.append({
                    'window_id': wid,
                    'k': test_k,
                    'silhouette_score': metrics['silhouette_score'],
                    'davies_bouldin_score': metrics['davies_bouldin_score'],
                    'calinski_harabasz_score': metrics['calinski_harabasz_score'],
                    'internal_edge_ratio': top_metrics['internal_edge_ratio'],
                    'modularity': mod
                })
                
        pd.DataFrame(k_sel_results).to_csv(out_tables_dir / "spectral_k_selection.csv", index=False)
        pd.DataFrame(eig_results).to_csv(out_tables_dir / "spectral_eigenvalues.csv", index=False)
        pd.DataFrame(eigengap_results).to_csv(out_tables_dir / "spectral_eigengap.csv", index=False)
        print("Saved K-selection and Eigenvalue diagnostics.")
        
    # ---------------------------------------------------------
    # MAIN CLUSTERING
    # ---------------------------------------------------------
    print(f"Running Spectral Clustering (K={spectral_cfg['n_clusters']}) for all graphs...")
    
    assignments = []
    cluster_stats = []
    metrics_list = []
    topology_list = []
    conductance_list = []
    
    vids_history = {} 
    labels_history = {}
    
    for wid in windows:
        graph_path = graphs_dir / f"graph_{wid:04d}.gpickle"
        if not graph_path.exists(): continue
        
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
            
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            vids_history[wid] = []
            labels_history[wid] = []
            continue
            
        vids = list(G.nodes())
        coords = []
        for n in vids:
            d = G.nodes[n]
            coords.append([d['x'], d['y'], d['speed']])
            
        X = np.array(coords)
        A = construct_adjacency_from_graph(G, vids)
        
        if n_nodes < spectral_cfg['n_clusters']:
            print(f"Warning: Window {wid} has {n_nodes} nodes, less than K={spectral_cfg['n_clusters']}")
            
        labels = run_spectral_clustering(A, n_clusters=spectral_cfg['n_clusters'], 
                                         random_state=spectral_cfg['random_state'],
                                         n_init=spectral_cfg['n_init'], 
                                         assign_labels=spectral_cfg['assign_labels'])
                                         
        vids_history[wid] = vids
        labels_history[wid] = labels
        
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
            
        # Cluster Statistics & Conductance
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_sz = counts.min()
        max_sz = counts.max()
        mean_sz = counts.mean()
        std_sz = counts.std() if len(counts) > 1 else 0.0
        
        avg_cond = []
        for lbl, count in zip(unique_labels, counts):
            idx = (labels == lbl)
            c_x = X[idx, 0].mean()
            c_y = X[idx, 1].mean()
            c_speed = X[idx, 2].mean()
            
            c_nodes = [vids[i] for i in range(len(vids)) if labels[i] == lbl]
            c_subgraph = G.subgraph(c_nodes)
            
            internal_edges = c_subgraph.number_of_edges()
            external_edges = 0
            for u in c_nodes:
                for v in G.neighbors(u):
                    if v not in c_nodes:
                        external_edges += 1
                        
            cond = calculate_conductance(G, c_nodes)
            if not np.isnan(cond):
                avg_cond.append(cond)
                
            conductance_list.append({
                'window_id': wid,
                'cluster': lbl,
                'cluster_size': count,
                'conductance': cond
            })
            
            density = (2 * internal_edges) / (count * (count - 1)) if count > 1 else 0.0
            
            cluster_stats.append({
                'window_id': wid,
                'cluster': lbl,
                'cluster_size': count,
                'mean_x': c_x,
                'mean_y': c_y,
                'mean_speed': c_speed,
                'internal_edges': internal_edges,
                'external_edges': external_edges,
                'internal_edge_density': density
            })
            
        # Feature metrics (Post-hoc evaluation)
        X_scaled = scale_features(X, method=scale_cfg['method'])
        c_metrics = calculate_clustering_metrics(X_scaled, labels)
        c_metrics['window_id'] = wid
        metrics_list.append(c_metrics)
        
        # Topology metrics
        t_metrics = calculate_topology_metrics(G, labels, vids)
        t_metrics['window_id'] = wid
        t_metrics['n_clusters'] = len(unique_labels)
        t_metrics['total_nodes'] = n_nodes
        t_metrics['modularity'] = calculate_modularity(G, labels, vids)
        t_metrics['average_cluster_conductance'] = np.mean(avg_cond) if avg_cond else np.nan
        topology_list.append(t_metrics)
        
    # Save Assignments
    assign_df = pd.DataFrame(assignments)
    assign_df.to_csv(out_clustering_dir / "spectral_assignments.csv", index=False)
    
    # Save Tables
    c_stats_df = pd.DataFrame(cluster_stats)
    c_stats_df.to_csv(out_tables_dir / "spectral_cluster_statistics.csv", index=False)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_tables_dir / "spectral_metrics.csv", index=False)
    
    top_df = pd.DataFrame(topology_list)
    # Reorder columns
    t_cols = ['window_id', 'n_clusters', 'total_nodes', 'total_edges', 'internal_edges', 'external_edges', 
              'internal_edge_ratio', 'modularity', 'average_cluster_conductance']
    top_df = top_df[[c for c in t_cols if c in top_df.columns]]
    top_df.to_csv(out_tables_dir / "spectral_topology_metrics.csv", index=False)
    
    pd.DataFrame(conductance_list).to_csv(out_tables_dir / "spectral_conductance.csv", index=False)
    
    # ---------------------------------------------------------
    # TEMPORAL OVERLAP
    # ---------------------------------------------------------
    temporal_overlaps = []
    vehicle_overlaps = []
    for i in range(len(windows) - 1):
        wid_curr = windows[i]
        wid_next = windows[i+1]
        
        vids_curr = vids_history[wid_curr]
        vids_next = vids_history[wid_next]
        labels_curr = labels_history[wid_curr]
        labels_next = labels_history[wid_next]
        
        overlap = calculate_temporal_overlap(vids_curr, vids_next)
        vehicle_overlaps.append({
            'window_id': wid_curr,
            'next_window_id': wid_next,
            'common_vehicle_count': overlap['common_vehicle_count'],
            'current_vehicle_count': overlap['total_current_vehicles'],
            'next_vehicle_count': overlap['total_next_vehicles']
        })
        
        # Jaccard for clusters
        unique_c_curr = set(labels_curr)
        unique_c_next = set(labels_next)
        
        for c1 in unique_c_curr:
            c1_nodes = set([vids_curr[j] for j in range(len(vids_curr)) if labels_curr[j] == c1])
            for c2 in unique_c_next:
                c2_nodes = set([vids_next[j] for j in range(len(vids_next)) if labels_next[j] == c2])
                
                intersection = c1_nodes.intersection(c2_nodes)
                union = c1_nodes.union(c2_nodes)
                jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
                
                temporal_overlaps.append({
                    'window_id': wid_curr,
                    'next_window_id': wid_next,
                    'cluster_t': c1,
                    'cluster_t1': c2,
                    'common_vehicle_count': len(intersection),
                    'jaccard_similarity': jaccard
                })
                
    pd.DataFrame(temporal_overlaps).to_csv(out_tables_dir / "spectral_temporal_overlap.csv", index=False)
    pd.DataFrame(vehicle_overlaps).to_csv(out_tables_dir / "spectral_vehicle_overlap.csv", index=False)
    
    # ---------------------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------------------
    print("Generating visualizations...")
    for wid in rep_windows:
        w_df = assign_df[assign_df['window_id'] == wid]
        if w_df.empty: continue
        
        plt.figure(figsize=(10, 8))
        
        graph_path = graphs_dir / f"graph_{wid:04d}.gpickle"
        if graph_path.exists():
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            pos = {n: (d['x'], d['y']) for n, d in G.nodes(data=True)}
            nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='gray')
            
        scatter = plt.scatter(w_df['x'], w_df['y'], c=w_df['cluster'], cmap='tab10', s=30, alpha=0.8)
        plt.title(f"Spectral Clusters (Window {wid}, K={spectral_cfg['n_clusters']})")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_fig_dir / f"spectral_window_{wid:04d}.png")
        plt.close()
        
    plt.figure(figsize=(10, 6))
    plt.hist(assign_df.groupby(['window_id', 'cluster']).size(), bins=30, edgecolor='black', alpha=0.7)
    plt.title('Spectral Cluster Size Distribution (All Windows)')
    plt.xlabel('Cluster Size (Number of Vehicles)')
    plt.ylabel('Frequency')
    plt.savefig(out_fig_dir / "cluster_size_distribution.png")
    plt.close()
    
    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    cluster_means = c_stats_df.groupby('window_id')['cluster_size'].agg(['mean', 'std'])
    
    summary = {
        'number_of_windows': len(windows),
        'n_clusters': spectral_cfg['n_clusters'],
        'average_silhouette': metrics_df['silhouette_score'].mean(),
        'average_davies_bouldin': metrics_df['davies_bouldin_score'].mean(),
        'average_calinski_harabasz': metrics_df['calinski_harabasz_score'].mean(),
        'average_modularity': top_df['modularity'].mean(),
        'average_internal_edge_ratio': top_df['internal_edge_ratio'].mean(),
        'average_cluster_size': cluster_means['mean'].mean(),
        'average_cluster_size_std': cluster_means['std'].mean(),
        'average_conductance': top_df['average_cluster_conductance'].mean()
    }
    pd.DataFrame([summary]).to_csv(out_tables_dir / "spectral_summary.csv", index=False)
    
    end_time = time.time()
    
    print("\n==========================================")
    print("PHASE 5 SUMMARY")
    print("==========================================")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Average time per graph: {(end_time - start_time) / len(windows):.4f} seconds")
    print(f"Number of graphs processed: {len(windows)}")
    print(f"Number of vehicles processed: {len(assign_df)}")
    print(f"K value used: {spectral_cfg['n_clusters']}")
    print(f"Adjacency representation: Unweighted Communication Topology")
    print(f"Laplacian representation: Normalized Symmetric (L_sym)")
    print(f"\nAverage silhouette: {summary['average_silhouette']:.4f}")
    print(f"Average Davies-Bouldin: {summary['average_davies_bouldin']:.4f}")
    print(f"Average Calinski-Harabasz: {summary['average_calinski_harabasz']:.4f}")
    print(f"Average modularity: {summary['average_modularity']:.4f}")
    print(f"Average internal edge ratio: {summary['average_internal_edge_ratio']:.4f}")
    print(f"Average cluster size: {summary['average_cluster_size']:.2f}")
    print(f"Average cluster-size std: {summary['average_cluster_size_std']:.2f}")
    print(f"Average conductance: {summary['average_conductance']:.4f}")
    print("==========================================")
    print("\nPhase 5 complete.")

if __name__ == "__main__":
    run_spectral_pipeline()
