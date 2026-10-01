import os
import yaml
import time
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import torch
import torch_geometric
from torch_geometric.utils import negative_sampling

from src.models.gae import GAEEncoder, VANETGraphAutoencoder
from src.models.kmeans_baseline import scale_features, run_kmeans
from src.evaluation.clustering_metrics import calculate_clustering_metrics
from src.evaluation.topology_metrics import calculate_topology_metrics, calculate_modularity, calculate_conductance
from src.evaluation.temporal_metrics import calculate_temporal_overlap

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_gae_pipeline(mode="full"):
    print("==========================================")
    if mode == "test":
        print("PHASE 6 GAE TEST MODE")
    elif mode == "benchmark":
        print("PHASE 6 GAE BENCHMARK MODE")
    else:
        print("--- PHASE 6: GRAPH AUTOENCODER (GAE) ---")
    print("==========================================")
    
    start_time = time.time()
    
    config = load_config()
    gae_cfg = config['clustering']['gae']
    scale_cfg = config['clustering']['scaling']
    
    # Thread Control
    cpu_threads = gae_cfg.get('cpu_threads', 'auto')
    if cpu_threads != 'auto':
        torch.set_num_threads(int(cpu_threads))
        print(f"Set CPU threads to: {cpu_threads}")
    
    # Random seeds
    seed = gae_cfg.get('seed', gae_cfg.get('random_seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device.type.upper()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyG version: {torch_geometric.__version__}")
    print("------------------------------------------")
    
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
        
    if mode == "test" or mode == "benchmark":
        windows = windows[:gae_cfg.get('test_graphs', 1)]
        epochs_limit = gae_cfg.get('test_epochs', 10)
    else:
        epochs_limit = gae_cfg['epochs']
        
    print(f"Loaded {len(windows)} graph snapshots.")
    
    out_clustering_dir = Path("data/processed/clustering/gae")
    out_clustering_dir.mkdir(parents=True, exist_ok=True)
    
    out_embeddings_dir = Path("data/processed/embeddings/gae")
    out_embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    out_tables_dir = Path("results/tables")
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    
    out_fig_dir = Path("results/figures/gae")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    
    out_models_dir = Path("results/models/gae")
    out_models_dir.mkdir(parents=True, exist_ok=True)
    
    rep_indices = [0, len(windows) // 4, len(windows) // 2, 3 * len(windows) // 4, len(windows) - 1]
    rep_windows = sorted(list(set([windows[i] for i in rep_indices if i < len(windows)])))
    
    training_loss_records = []
    reconstruction_metrics = []
    embedding_stats = []
    
    assignments = []
    cluster_stats = []
    metrics_list = []
    topology_list = []
    conductance_list = []
    
    vids_history = {} 
    labels_history = {}
    epochs_completed_list = []
    
    for idx, wid in enumerate(windows):
        g_start_time = time.time()
        
        # Benchmarking times
        t_load = 0; t_prep = 0; t_fwd = 0; t_bwd = 0; t_recon = 0
        
        t0 = time.time()
        graph_path = graphs_dir / f"graph_{wid:04d}.gpickle"
        if not graph_path.exists(): 
            print(f"Warning: Graph {wid} not found.")
            continue
            
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
            
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            vids_history[wid] = []
            labels_history[wid] = []
            continue
            
        t_load += time.time() - t0
        
        if mode != "benchmark":
            print(f"\n[{idx+1}/{len(windows)}] Window {wid}")
            print(f"    Nodes: {n_nodes}")
            print(f"    Edges: {G.number_of_edges()}")
        elif mode == "benchmark":
            print(f"Graph: window {wid}")
            print(f"Nodes: {n_nodes}")
            print(f"Edges: {G.number_of_edges()}")
            print(f"Feature dimension: 3")
            print(f"Hidden dimension: {gae_cfg['hidden_channels']}")
            print(f"Latent dimension: {gae_cfg['latent_channels']}")
            print(f"Epochs: {epochs_limit}")
            print(f"Device: {device.type.upper()}")
            print("------------------------------------------")
            print("Training")
            print("------------------------------------------")
        
        t0 = time.time()
        vids = list(G.nodes())
        coords = [[G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['speed']] for n in vids]
            
        X_orig = np.array(coords)
        X_scaled = scale_features(X_orig, method=scale_cfg['method'])
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        # Validate features
        if torch.isnan(x_tensor).any() or torch.isinf(x_tensor).any():
            print(f"    ERROR: Invalid feature values in window {wid}. Skipping.")
            continue
        
        vid_to_idx = {vid: i for i, vid in enumerate(vids)}
        edges = [(vid_to_idx[u], vid_to_idx[v]) for u, v in G.edges()]
        
        edge_index_list = []
        for u, v in edges:
            edge_index_list.append([u, v])
            edge_index_list.append([v, u])
            
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            
        if edge_index.size(1) > 0 and (edge_index.min() < 0 or edge_index.max() >= n_nodes):
            print(f"    ERROR: Invalid edge indices in window {wid}. Skipping.")
            continue
            
        encoder = GAEEncoder(
            in_channels=3, 
            hidden_channels=gae_cfg['hidden_channels'], 
            out_channels=gae_cfg['latent_channels']
        )
        model = VANETGraphAutoencoder(encoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=gae_cfg['learning_rate'], weight_decay=gae_cfg['weight_decay'])
        t_prep += time.time() - t0
        
        best_loss = float('inf')
        best_state = None
        best_epoch = 0
        patience_counter = 0
        epochs_completed = 0
        
        neg_ratio = gae_cfg.get('negative_sampling_ratio', 1.0)
        num_neg_samples = int(edge_index.size(1) * neg_ratio)
        
        if mode != "benchmark":
            print("    Training...")
        
        for epoch in range(1, epochs_limit + 1):
            t_ep_start = time.time()
            model.train()
            optimizer.zero_grad()
            
            t0 = time.time()
            z = model.encode(x_tensor, edge_index)
            t_fwd += time.time() - t0
            
            if torch.isnan(z).any() or torch.isinf(z).any():
                print(f"    Warning: NaN/Inf embeddings in window {wid} at epoch {epoch}. Stopping.")
                break
                
            t0 = time.time()
            neg_edge_index = negative_sampling(edge_index, num_nodes=n_nodes, num_neg_samples=num_neg_samples)
            loss = model.recon_loss(z, edge_index, neg_edge_index)
            t_recon += time.time() - t0
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    Warning: NaN/Inf loss in window {wid} at epoch {epoch}. Stopping.")
                break
                
            t0 = time.time()
            loss.backward()
            optimizer.step()
            t_bwd += time.time() - t0
            
            loss_val = loss.item()
            training_loss_records.append({'window_id': wid, 'epoch': epoch, 'loss': loss_val})
            epochs_completed = epoch
            
            if mode == "test" or mode == "benchmark":
                print(f"Epoch {epoch}/{epochs_limit}")
                print(f"Loss: {loss_val:.4f}")
            
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= gae_cfg['patience']:
                break
                
        epochs_completed_list.append(epochs_completed)
        
        if best_state is not None:
            model.load_state_dict(best_state)
            
        g_elapsed = time.time() - g_start_time
        if mode != "benchmark":
            print(f"    Epochs: {epochs_completed} | Best Epoch: {best_epoch} | Loss: {best_loss:.4f} | Time: {g_elapsed:.2f}s")
        
        if g_elapsed > 10.0 and mode != "benchmark":
            print(f"    WARNING: unusually slow graph. window_id={wid}, nodes={n_nodes}, edges={G.number_of_edges()}, time={g_elapsed:.2f}s")
            
        if mode == "benchmark":
            print("------------------------------------------")
            print("PERFORMANCE PROFILE")
            print("------------------------------------------")
            print(f"Graph loading:      {t_load:.4f} sec")
            print(f"Tensor preparation: {t_prep:.4f} sec")
            print(f"Forward:            {t_fwd:.4f} sec")
            print(f"Backward:           {t_bwd:.4f} sec")
            print(f"Reconstruction:     {t_recon:.4f} sec")
            print(f"Total:              {g_elapsed:.4f} sec")
            return
        
        if gae_cfg['save_checkpoints'] and wid in rep_windows:
            torch.save(model.state_dict(), out_models_dir / f"gae_window_{wid:04d}.pt")
            
        model.eval()
        with torch.no_grad():
            z = model.encode(x_tensor, edge_index)
            # Evaluate reconstruction using decode_all
            adj_pred = model.decode_all(z)
            
        z_np = z.cpu().numpy()
        adj_pred_np = adj_pred.cpu().numpy()
        
        emb_df = pd.DataFrame(z_np, columns=[f"z_{i}" for i in range(gae_cfg['latent_channels'])])
        emb_df.insert(0, 'vehicle_id', vids)
        emb_df.insert(0, 'window_id', wid)
        emb_df.to_csv(out_embeddings_dir / f"gae_window_{wid:04d}.csv", index=False)
        
        for i in range(gae_cfg['latent_channels']):
            embedding_stats.append({
                'window_id': wid, 'dimension': i,
                'mean': float(z_np[:, i].mean()), 'std': float(z_np[:, i].std()),
                'min': float(z_np[:, i].min()), 'max': float(z_np[:, i].max())
            })
            
        n = n_nodes
        if n > 1:
            adj_true_np = np.zeros((n, n))
            if edge_index.size(1) > 0:
                ei_np = edge_index.cpu().numpy()
                adj_true_np[ei_np[0], ei_np[1]] = 1.0
                
            mask = ~np.eye(n, dtype=bool)
            y_true = adj_true_np[mask]
            y_pred = adj_pred_np[mask]
            
            if y_true.sum() > 0 and y_true.sum() < len(y_true):
                roc = roc_auc_score(y_true, y_pred)
                ap = average_precision_score(y_true, y_pred)
            else:
                roc = np.nan
                ap = np.nan
                
            y_pred_bin = (y_pred >= 0.5).astype(int)
            true_pos = np.sum((y_true == 1) & (y_pred_bin == 1))
            false_pos = np.sum((y_true == 0) & (y_pred_bin == 1))
            false_neg = np.sum((y_true == 1) & (y_pred_bin == 0))
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            reconstruction_metrics.append({
                'window_id': wid, 'reconstruction_loss': best_loss,
                'roc_auc': roc, 'average_precision': ap,
                'reconstructed_edges': int(np.sum(y_pred_bin)), 'true_edges': int(np.sum(y_true)),
                'true_positives': int(true_pos), 'false_positives': int(false_pos), 'false_negatives': int(false_neg),
                'precision': precision, 'recall': recall, 'f1': f1
            })
        else:
            reconstruction_metrics.append({
                'window_id': wid, 'reconstruction_loss': best_loss, 'roc_auc': np.nan, 'average_precision': np.nan,
                'reconstructed_edges': 0, 'true_edges': 0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            })
            
        n_clusters_adj = min(gae_cfg['cluster_count'], n_nodes)
        if n_clusters_adj >= 2:
            labels, _ = run_kmeans(z_np, k=n_clusters_adj, random_state=seed, n_init=gae_cfg['n_init'])
        else:
            labels = np.zeros(n_nodes, dtype=int)
            
        vids_history[wid] = vids
        labels_history[wid] = labels
        
        for i, vid in enumerate(vids):
            assignments.append({
                'window_id': wid, 'vehicle_id': vid,
                'x': X_orig[i, 0], 'y': X_orig[i, 1], 'speed': X_orig[i, 2],
                'cluster': labels[i]
            })
            
        unique_labels, counts = np.unique(labels, return_counts=True)
        for lbl, count in zip(unique_labels, counts):
            idx = (labels == lbl)
            c_nodes = [vids[i] for i in range(len(vids)) if labels[i] == lbl]
            cond = calculate_conductance(G, c_nodes)
            if not np.isnan(cond):
                conductance_list.append({'window_id': wid, 'cluster': lbl, 'cluster_size': count, 'conductance': cond})
                
            cluster_stats.append({
                'window_id': wid, 'cluster': lbl, 'cluster_size': count,
                'mean_x': X_orig[idx, 0].mean(), 'mean_y': X_orig[idx, 1].mean(), 'mean_speed': X_orig[idx, 2].mean()
            })
            
        c_metrics = calculate_clustering_metrics(z_np, labels)
        c_metrics['window_id'] = wid
        c_metrics['evaluation_space'] = 'embedding'
        metrics_list.append(c_metrics)
        
        c_metrics_spatial = calculate_clustering_metrics(X_scaled, labels)
        c_metrics_spatial['window_id'] = wid
        c_metrics_spatial['evaluation_space'] = 'spatial'
        metrics_list.append(c_metrics_spatial)
        
        t_metrics = calculate_topology_metrics(G, labels, vids)
        t_metrics['window_id'] = wid
        t_metrics['n_clusters'] = len(unique_labels)
        t_metrics['modularity'] = calculate_modularity(G, labels, vids)
        topology_list.append(t_metrics)
        
    if mode == "test":
        print("GAE test successful.")
        return
        
    pd.DataFrame(training_loss_records).to_csv(out_tables_dir / "gae_training_loss.csv", index=False)
    pd.DataFrame(reconstruction_metrics).to_csv(out_tables_dir / "gae_reconstruction_metrics.csv", index=False)
    pd.DataFrame(embedding_stats).to_csv(out_tables_dir / "gae_embedding_statistics.csv", index=False)
    pd.DataFrame(assignments).to_csv(out_clustering_dir / "gae_assignments.csv", index=False)
    pd.DataFrame(cluster_stats).to_csv(out_tables_dir / "gae_cluster_statistics.csv", index=False)
    pd.DataFrame(metrics_list).to_csv(out_tables_dir / "gae_metrics.csv", index=False)
    pd.DataFrame(topology_list).to_csv(out_tables_dir / "gae_topology_metrics.csv", index=False)
    pd.DataFrame(conductance_list).to_csv(out_tables_dir / "gae_conductance.csv", index=False)
    
    temporal_overlaps = []
    vehicle_overlaps = []
    for i in range(len(windows) - 1):
        wid_curr = windows[i]
        wid_next = windows[i+1]
        
        vids_curr = vids_history.get(wid_curr, [])
        vids_next = vids_history.get(wid_next, [])
        labels_curr = labels_history.get(wid_curr, [])
        labels_next = labels_history.get(wid_next, [])
        
        overlap = calculate_temporal_overlap(vids_curr, vids_next)
        vehicle_overlaps.append({
            'window_id': wid_curr, 'next_window_id': wid_next,
            'common_vehicle_count': overlap['common_vehicle_count'],
            'current_vehicle_count': overlap['total_current_vehicles'],
            'next_vehicle_count': overlap['total_next_vehicles']
        })
        
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
                    'window_id': wid_curr, 'next_window_id': wid_next,
                    'cluster_t': c1, 'cluster_t1': c2,
                    'common_vehicle_count': len(intersection), 'jaccard_similarity': jaccard
                })
                
    pd.DataFrame(temporal_overlaps).to_csv(out_tables_dir / "gae_temporal_overlap.csv", index=False)
    pd.DataFrame(vehicle_overlaps).to_csv(out_tables_dir / "gae_vehicle_overlap.csv", index=False)
    
    print("\nGenerating visualizations...")
    assign_df = pd.DataFrame(assignments)
    
    if training_loss_records:
        first_rep = rep_windows[0]
        rep_loss = pd.DataFrame(training_loss_records)
        rep_loss = rep_loss[rep_loss['window_id'] == first_rep]
        if not rep_loss.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(rep_loss['epoch'], rep_loss['loss'])
            plt.title(f'GAE Training Loss (Window {first_rep})')
            plt.xlabel('Epoch')
            plt.ylabel('BCE Loss')
            plt.grid(True)
            plt.savefig(out_fig_dir / "gae_training_loss.png")
            plt.close()
            
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
        plt.title(f"GAE Clusters Spatial View (Window {wid})")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_fig_dir / f"gae_window_{wid:04d}.png")
        plt.close()
        
        emb_path = out_embeddings_dir / f"gae_window_{wid:04d}.csv"
        if emb_path.exists():
            emb_df = pd.read_csv(emb_path)
            z_cols = [c for c in emb_df.columns if c.startswith('z_')]
            if len(z_cols) >= 2:
                pca = PCA(n_components=2)
                z_2d = pca.fit_transform(emb_df[z_cols].values)
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=w_df['cluster'], cmap='tab10', s=30, alpha=0.8)
                plt.title(f"GAE Latent Embeddings (PCA) (Window {wid})")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.colorbar(scatter, label="Cluster ID")
                plt.grid(True, alpha=0.3)
                plt.savefig(out_fig_dir / f"gae_embedding_window_{wid:04d}.png")
                plt.close()
                
    c_stats_df = pd.DataFrame(cluster_stats)
    cluster_means = c_stats_df.groupby('window_id')['cluster_size'].agg(['mean', 'std'])
    metrics_df = pd.DataFrame(metrics_list)
    emb_metrics = metrics_df[metrics_df['evaluation_space'] == 'embedding']
    top_df = pd.DataFrame(topology_list)
    recon_df = pd.DataFrame(reconstruction_metrics)
    
    summary = {
        'number_of_windows': len(windows),
        'average_nodes': len(assign_df) / len(windows) if len(windows) > 0 else 0,
        'latent_dimension': gae_cfg['latent_channels'],
        'average_training_epochs': np.mean(epochs_completed_list),
        'average_reconstruction_loss': recon_df['reconstruction_loss'].mean(),
        'average_roc_auc': recon_df['roc_auc'].mean(),
        'average_average_precision': recon_df['average_precision'].mean(),
        'average_silhouette': emb_metrics['silhouette_score'].mean() if not emb_metrics.empty else np.nan,
        'average_davies_bouldin': emb_metrics['davies_bouldin_score'].mean() if not emb_metrics.empty else np.nan,
        'average_calinski_harabasz': emb_metrics['calinski_harabasz_score'].mean() if not emb_metrics.empty else np.nan,
        'average_modularity': top_df['modularity'].mean() if not top_df.empty else np.nan,
        'average_internal_edge_ratio': top_df['internal_edge_ratio'].mean() if not top_df.empty else np.nan,
        'average_cluster_size': cluster_means['mean'].mean() if not cluster_means.empty else np.nan,
        'average_cluster_size_std': cluster_means['std'].mean() if not cluster_means.empty else np.nan
    }
    pd.DataFrame([summary]).to_csv(out_tables_dir / "gae_summary.csv", index=False)
    
    end_time = time.time()
    
    print("\n==========================================")
    print("PHASE 6 SUMMARY (GAE)")
    print("==========================================")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Average time per graph: {(end_time - start_time) / len(windows):.4f} seconds")
    print(f"Number of graphs processed: {len(windows)}")
    print(f"Node feature dimension: 3 [x, y, speed]")
    print(f"GAE hidden dimension: {gae_cfg['hidden_channels']}")
    print(f"GAE latent dimension: {gae_cfg['latent_channels']}")
    print(f"Average epochs trained: {summary['average_training_epochs']:.1f}")
    print(f"Learning rate: {gae_cfg['learning_rate']}")
    print(f"Optimizer: Adam")
    print(f"\nAverage reconstruction loss: {summary['average_reconstruction_loss']:.4f}")
    print(f"Average ROC-AUC: {summary['average_roc_auc']:.4f}")
    print(f"Average Average Precision: {summary['average_average_precision']:.4f}")
    print(f"\nAverage Silhouette (Latent): {summary['average_silhouette']:.4f}")
    print(f"Average Davies-Bouldin (Latent): {summary['average_davies_bouldin']:.4f}")
    print(f"Average Calinski-Harabasz (Latent): {summary['average_calinski_harabasz']:.4f}")
    print(f"\nAverage modularity: {summary['average_modularity']:.4f}")
    print(f"Average internal edge ratio: {summary['average_internal_edge_ratio']:.4f}")
    print("==========================================")
    print("\nPhase 6 complete.")

if __name__ == "__main__":
    run_gae_pipeline()
