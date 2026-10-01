import os
import yaml
import time
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.kmeans_baseline import run_kmeans
from src.models.cluster_matching import match_clusters
from src.models.cluster_head import select_cluster_heads
from src.evaluation.clustering_metrics import calculate_clustering_metrics
from src.evaluation.topology_metrics import calculate_topology_metrics, calculate_modularity
from src.evaluation.temporal_metrics import calculate_temporal_overlap

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_stable_pipeline(mode="full"):
    print("==========================================")
    if mode == "test":
        print("PHASE 7: STABLE VANET CLUSTERING (TEST MODE)")
    else:
        print("PHASE 7: STABLE VANET CLUSTERING")
    print("==========================================")
    
    start_time = time.time()
    config = load_config()
    stable_cfg = config['clustering']['stable']
    gae_cfg = config['clustering']['gae']
    
    seed = gae_cfg.get('seed', 42)
    np.random.seed(seed)
    
    graphs_dir = Path("data/processed/graphs")
    meta_path = graphs_dir / "graph_metadata.csv"
    emb_dir = Path("data/processed/embeddings/gae")
    
    out_clustering_dir = Path("data/processed/clustering/stable_gae")
    out_clustering_dir.mkdir(parents=True, exist_ok=True)
    out_tables_dir = Path("results/tables")
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir = Path("results/figures/phase7")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    
    meta_df = pd.read_csv(meta_path)
    windows = meta_df['window_id'].sort_values().tolist()
    
    if mode == "test":
        windows = windows[:stable_cfg.get('test_graphs', 5)]
        
    num_windows = len(windows)
    num_transitions = max(0, num_windows - 1)
    
    print(f"Graph windows:    {num_windows}")
    print(f"Transitions:      {num_transitions}")
    print(f"Clustering:       K-Means on GAE embeddings")
    print(f"Latent dimension: {gae_cfg['latent_channels']}")
    print("Cluster-head selection: topology + mobility + temporal stability")
    print("------------------------------------------")
    
    # State tracking
    prev_clusters = {} # {global_cid: set(vids)}
    prev_heads = {}    # {global_cid: head_vid}
    
    head_lifetimes = {} # {vid: current_consecutive_lifetime}
    historical_lifetimes = []
    
    # Data collections
    all_assignments = []
    all_heads = []
    cluster_stats = []
    
    transition_metrics = []
    vehicle_stability_records = []
    
    overall_modularity = []
    overall_internal_edge_ratio = []
    overall_silhouette = []
    overall_davies_bouldin = []
    overall_calinski_harabasz = []
    
    for idx, wid in enumerate(windows):
        graph_path = graphs_dir / f"graph_{wid:04d}.gpickle"
        emb_path = emb_dir / f"gae_window_{wid:04d}.csv"
        
        if not graph_path.exists() or not emb_path.exists():
            continue
            
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
            
        emb_df = pd.read_csv(emb_path)
        vids = emb_df['vehicle_id'].values
        z_cols = [c for c in emb_df.columns if c.startswith('z_')]
        Z = emb_df[z_cols].values
        
        n_nodes = len(vids)
        if n_nodes == 0:
            prev_clusters = {}
            prev_heads = {}
            continue
            
        k = min(stable_cfg['k'], n_nodes)
        
        # 1. K-Means Clustering on GAE Embeddings
        if k >= 2:
            labels, _ = run_kmeans(Z, k=k, random_state=seed, n_init=gae_cfg['n_init'])
        else:
            labels = np.zeros(n_nodes, dtype=int)
            
        local_clusters = {}
        for i, vid in enumerate(vids):
            lbl = labels[i]
            if lbl not in local_clusters:
                local_clusters[lbl] = set()
            local_clusters[lbl].add(vid)
            
        # 2. Cluster Matching
        mapping = match_clusters(prev_clusters, local_clusters)
        
        # Build current clusters with global IDs
        curr_clusters = {}
        for local_cid, members in local_clusters.items():
            global_cid = mapping[local_cid]
            curr_clusters[global_cid] = members
            
        # 3. Cluster Head Selection
        curr_heads, head_scores = select_cluster_heads(
            G=G,
            clusters=curr_clusters,
            prev_heads=prev_heads,
            prev_clusters=prev_clusters,
            weights=stable_cfg
        )
        
        # Track head lifetime
        for cid, head in curr_heads.items():
            if cid in prev_heads and prev_heads[cid] == head:
                head_lifetimes[head] = head_lifetimes.get(head, 1) + 1
            else:
                if cid in prev_heads:
                    old_head = prev_heads[cid]
                    if old_head in head_lifetimes:
                        historical_lifetimes.append({
                            'vehicle_id': old_head,
                            'lifetime': head_lifetimes[old_head]
                        })
                        del head_lifetimes[old_head]
                head_lifetimes[head] = 1
                
        # 4. Temporal Evaluation (if not first window)
        if prev_clusters:
            jaccard_sum = 0.0
            retained_vehicles = 0
            both_window_vehicles = 0
            head_changes = 0
            comparable_clusters = 0
            
            for cid, members in curr_clusters.items():
                if cid in prev_clusters:
                    comparable_clusters += 1
                    p_members = prev_clusters[cid]
                    intersection = len(members.intersection(p_members))
                    union = len(members.union(p_members))
                    jaccard = intersection / union if union > 0 else 0
                    jaccard_sum += jaccard
                    
                    if curr_heads[cid] != prev_heads[cid]:
                        head_changes += 1
                        
            # Vehicle retention
            curr_vids = set(vids)
            prev_vids = set()
            for p_members in prev_clusters.values():
                prev_vids.update(p_members)
                
            common_vids = curr_vids.intersection(prev_vids)
            both_window_vehicles = len(common_vids)
            
            for vid in common_vids:
                # Find which cluster it was in before
                prev_c = None
                for c, m in prev_clusters.items():
                    if vid in m: prev_c = c; break
                # Find which cluster it is in now
                curr_c = None
                for c, m in curr_clusters.items():
                    if vid in m: curr_c = c; break
                    
                if prev_c == curr_c:
                    retained_vehicles += 1
                    
            avg_jaccard = jaccard_sum / comparable_clusters if comparable_clusters > 0 else 0
            retention_rate = retained_vehicles / both_window_vehicles if both_window_vehicles > 0 else 0
            head_change_rate = head_changes / comparable_clusters if comparable_clusters > 0 else 0
            
            transition_metrics.append({
                'window_t_minus_1': windows[idx-1],
                'window_t': wid,
                'avg_jaccard': avg_jaccard,
                'vehicle_retention_rate': retention_rate,
                'head_change_rate': head_change_rate,
                'comparable_clusters': comparable_clusters
            })
            
            if mode != "full" or idx % 10 == 0 or idx == num_windows - 1:
                print(f"[Transition {idx}/{num_transitions}] Window {windows[idx-1]} -> {wid}")
                print(f"    Avg Jaccard: {avg_jaccard:.4f} | Retention: {retention_rate:.4f} | Head Change Rate: {head_change_rate:.4f}")
        else:
            if mode != "full" or idx == 0:
                print(f"[1/{num_windows}] Window {wid} (Initial)")
                
        # Global assignments for this window
        global_labels = np.zeros(n_nodes, dtype=int)
        for i, vid in enumerate(vids):
            for global_cid, members in curr_clusters.items():
                if vid in members:
                    global_labels[i] = global_cid
                    break
                    
        # Metrics
        if k >= 2:
            c_metrics = calculate_clustering_metrics(Z, global_labels)
            overall_silhouette.append(c_metrics.get('silhouette_score', np.nan))
            overall_davies_bouldin.append(c_metrics.get('davies_bouldin_score', np.nan))
            overall_calinski_harabasz.append(c_metrics.get('calinski_harabasz_score', np.nan))
            
        t_metrics = calculate_topology_metrics(G, global_labels, vids)
        overall_internal_edge_ratio.append(t_metrics.get('internal_edge_ratio', np.nan))
        overall_modularity.append(calculate_modularity(G, global_labels, vids))
        
        # Save output records
        for cid, head_vid in curr_heads.items():
            all_heads.append({
                'window_id': wid,
                'cluster_id': cid,
                'head_vehicle_id': head_vid,
                'head_score': head_scores[head_vid],
                'degree': G.degree(head_vid),
                'speed': G.nodes[head_vid]['speed'],
                'cluster_size': len(curr_clusters[cid])
            })
            
        for i, vid in enumerate(vids):
            cid = global_labels[i]
            all_assignments.append({
                'window_id': wid,
                'vehicle_id': vid,
                'cluster_id': cid,
                'is_head': int(vid == curr_heads[cid]),
                'x': G.nodes[vid]['x'],
                'y': G.nodes[vid]['y'],
                'speed': G.nodes[vid]['speed']
            })
            
        for cid, members in curr_clusters.items():
            cluster_stats.append({
                'window_id': wid,
                'cluster_id': cid,
                'cluster_size': len(members),
                'head_vehicle_id': curr_heads[cid]
            })
            
        prev_clusters = curr_clusters
        prev_heads = curr_heads
        
    # Finalize lifetimes
    for vid, life in head_lifetimes.items():
        historical_lifetimes.append({'vehicle_id': vid, 'lifetime': life})
        
    # Write files
    pd.DataFrame(all_assignments).to_csv(out_clustering_dir / "cluster_assignments.csv", index=False)
    pd.DataFrame(all_heads).to_csv(out_clustering_dir / "cluster_heads.csv", index=False)
    pd.DataFrame(historical_lifetimes).to_csv(out_clustering_dir / "cluster_head_lifetime.csv", index=False)
    
    t_metrics_df = pd.DataFrame(transition_metrics)
    t_metrics_df.to_csv(out_tables_dir / "phase7_transition_metrics.csv", index=False)
    pd.DataFrame(cluster_stats).to_csv(out_tables_dir / "phase7_cluster_statistics.csv", index=False)
    pd.DataFrame(all_heads).to_csv(out_tables_dir / "phase7_cluster_heads.csv", index=False)
    pd.DataFrame(historical_lifetimes).to_csv(out_tables_dir / "phase7_head_lifetime.csv", index=False)
    
    # Plotting
    if not t_metrics_df.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(t_metrics_df['window_t'], t_metrics_df['avg_jaccard'], label="Jaccard")
        plt.plot(t_metrics_df['window_t'], t_metrics_df['vehicle_retention_rate'], label="Vehicle Retention")
        plt.title('Cluster Stability Over Time')
        plt.xlabel('Time Window')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_fig_dir / "stability_over_time.png")
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(t_metrics_df['window_t'], t_metrics_df['head_change_rate'] * t_metrics_df['comparable_clusters'], label="Head Changes", color='red')
        plt.title('Cluster-Head Changes Over Time')
        plt.xlabel('Time Window')
        plt.ylabel('Number of Changed Heads')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_fig_dir / "head_changes_over_time.png")
        plt.close()
        
    if historical_lifetimes:
        hl_df = pd.DataFrame(historical_lifetimes)
        plt.figure(figsize=(8, 5))
        plt.hist(hl_df['lifetime'], bins=range(1, hl_df['lifetime'].max() + 2), align='left', rwidth=0.8)
        plt.title('Cluster-Head Lifetime Distribution')
        plt.xlabel('Lifetime (Consecutive Windows)')
        plt.ylabel('Number of Cluster Heads')
        plt.grid(True, alpha=0.3)
        plt.savefig(out_fig_dir / "head_lifetime_distribution.png")
        plt.close()
        
        avg_head_lifetime = hl_df['lifetime'].mean()
        max_head_lifetime = hl_df['lifetime'].max()
        med_head_lifetime = hl_df['lifetime'].median()
    else:
        avg_head_lifetime = max_head_lifetime = med_head_lifetime = 0.0
        
    c_stats_df = pd.DataFrame(cluster_stats)
    if not c_stats_df.empty:
        c_size_mean = c_stats_df.groupby('window_id')['cluster_size'].mean()
        plt.figure(figsize=(10, 5))
        plt.plot(c_size_mean.index, c_size_mean.values)
        plt.title('Average Cluster Size Over Time')
        plt.xlabel('Time Window')
        plt.ylabel('Cluster Size')
        plt.grid(True)
        plt.savefig(out_fig_dir / "cluster_size_over_time.png")
        plt.close()
        
    avg_jaccard = t_metrics_df['avg_jaccard'].mean() if not t_metrics_df.empty else 0.0
    avg_retention = t_metrics_df['vehicle_retention_rate'].mean() if not t_metrics_df.empty else 0.0
    avg_head_change_rate = t_metrics_df['head_change_rate'].mean() if not t_metrics_df.empty else 0.0
    avg_head_retention = 1.0 - avg_head_change_rate
    
    summary = {
        'windows': num_windows,
        'transitions': num_transitions,
        'average_jaccard': avg_jaccard,
        'average_vehicle_retention': avg_retention,
        'head_change_rate': avg_head_change_rate,
        'head_retention': avg_head_retention,
        'average_head_lifetime': avg_head_lifetime,
        'maximum_head_lifetime': max_head_lifetime,
        'average_modularity': np.nanmean(overall_modularity) if overall_modularity else 0.0,
        'average_internal_edge_ratio': np.nanmean(overall_internal_edge_ratio) if overall_internal_edge_ratio else 0.0,
        'average_silhouette': np.nanmean(overall_silhouette) if overall_silhouette else 0.0,
        'average_davies_bouldin': np.nanmean(overall_davies_bouldin) if overall_davies_bouldin else 0.0,
        'average_calinski_harabasz': np.nanmean(overall_calinski_harabasz) if overall_calinski_harabasz else 0.0,
    }
    pd.DataFrame([summary]).to_csv(out_tables_dir / "phase7_summary.csv", index=False)
    
    print("\n==========================================")
    print("PHASE 7 SUMMARY")
    print("==========================================")
    print(f"Graph windows:    {num_windows}")
    print(f"Transitions:      {num_transitions}")
    print(f"Number of clusters: {stable_cfg['k']}")
    print("------------------------------------------")
    print("TEMPORAL STABILITY")
    print("------------------------------------------")
    print(f"Average Jaccard:           {summary['average_jaccard']:.4f}")
    print(f"Average vehicle retention: {summary['average_vehicle_retention']:.4f}")
    print(f"Cluster change rate:       {(1 - summary['average_vehicle_retention']):.4f}")
    print("------------------------------------------")
    print("CLUSTER HEADS")
    print("------------------------------------------")
    print(f"Head change rate:          {summary['head_change_rate']:.4f}")
    print(f"Head retention:            {summary['head_retention']:.4f}")
    print(f"Average head lifetime:     {summary['average_head_lifetime']:.2f}")
    print(f"Maximum head lifetime:     {summary['maximum_head_lifetime']:.2f}")
    print(f"Median head lifetime:      {med_head_lifetime:.2f}")
    print("------------------------------------------")
    print("TOPOLOGY")
    print("------------------------------------------")
    print(f"Average modularity:          {summary['average_modularity']:.4f}")
    print(f"Average internal edge ratio: {summary['average_internal_edge_ratio']:.4f}")
    print("------------------------------------------")
    print("CLUSTERING")
    print("------------------------------------------")
    print(f"Average silhouette:        {summary['average_silhouette']:.4f}")
    print(f"Average Davies-Bouldin:    {summary['average_davies_bouldin']:.4f}")
    print(f"Average Calinski-Harabasz: {summary['average_calinski_harabasz']:.4f}")
    print("==========================================")
    
    if mode == "test":
        print("\nPhase 7 test mode PASSED")

if __name__ == "__main__":
    run_stable_pipeline()
