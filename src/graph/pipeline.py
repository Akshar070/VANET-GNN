import os
import yaml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from pathlib import Path

from src.data.windowing import create_time_windows
from src.graph.builder import construct_dynamic_graphs

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_graph_statistics(metrics_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Nodes and edges per window
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['window_id'], metrics_df['node_count'], label='Nodes')
    plt.plot(metrics_df['window_id'], metrics_df['edge_count'], label='Edges')
    plt.title('Nodes and Edges over Time')
    plt.xlabel('Window ID')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(output_dir / 'nodes_edges_per_window.png')
    plt.close()
    
    # 2. Average degree
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['window_id'], metrics_df['average_degree'])
    plt.title('Average Degree over Time')
    plt.xlabel('Window ID')
    plt.ylabel('Average Degree')
    plt.savefig(output_dir / 'average_degree_over_time.png')
    plt.close()
    
    # 3. Connected components
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['window_id'], metrics_df['connected_components'])
    plt.title('Connected Components over Time')
    plt.xlabel('Window ID')
    plt.ylabel('Number of Components')
    plt.savefig(output_dir / 'connected_components_over_time.png')
    plt.close()
    
    # 4. Graph Density
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['window_id'], metrics_df['graph_density'])
    plt.title('Graph Density over Time')
    plt.xlabel('Window ID')
    plt.ylabel('Density')
    plt.savefig(output_dir / 'graph_density_over_time.png')
    plt.close()

def plot_representative_graphs(graphs_dir, output_dir, metrics_df):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    windows = metrics_df['window_id'].tolist()
    if not windows:
        return
        
    # Select first, early, middle, late, final
    indices = [
        0,
        len(windows) // 4,
        len(windows) // 2,
        3 * len(windows) // 4,
        len(windows) - 1
    ]
    # Remove duplicates if few windows
    indices = sorted(list(set([i for i in indices if i < len(windows)])))
    
    for idx in indices:
        wid = windows[idx]
        graph_path = Path(graphs_dir) / f"graph_{wid:04d}.gpickle"
        if not graph_path.exists():
            continue
            
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
            
        plt.figure(figsize=(12, 12))
        pos = {n: (d['x'], d['y']) for n, d in G.nodes(data=True)}
        
        # Plot edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
        # Plot nodes
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue', alpha=0.7)
        
        plt.title(f"Dynamic VANET Graph Snapshot (Window {wid})")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.grid(True, alpha=0.3)
        # Add a custom axis to show spatial distribution clearly
        ax = plt.gca()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.savefig(output_dir / f"graph_window_{wid:04d}.png")
        plt.close()

def run_graph_pipeline():
    print("--- PHASE 3: DYNAMIC VANET GRAPH CONSTRUCTION ---")
    start_time = time.time()
    
    config = load_config()
    data_cfg = config['data']
    window_cfg = config['window']
    graph_cfg = config['graph']
    
    input_file = Path(data_cfg['processed_path'])
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run preprocess first.")
        return
        
    df = pd.read_csv(input_file)
    
    window_size = window_cfg['size_seconds']
    step_size = window_cfg['step_seconds']
    comm_range = graph_cfg['communication_range_m']
    
    windows_dir = Path("data/processed/windows")
    graphs_dir = Path("data/processed/graphs")
    
    # 1. Create Time Windows
    print(f"Generating time windows (size={window_size}s, step={step_size}s)...")
    metadata_df, snapshots_df = create_time_windows(df, window_size, step_size, windows_dir)
    
    if snapshots_df.empty:
        print("No snapshots generated. Exiting.")
        return
        
    # 2. Construct Dynamic Graphs
    metrics_df = construct_dynamic_graphs(snapshots_df, metadata_df, graphs_dir, comm_range)
    
    # 3. Generate Visualizations
    print("Generating visualizations...")
    plot_graph_statistics(metrics_df, "results/figures/graph_statistics")
    plot_representative_graphs(graphs_dir, "results/figures/graphs", metrics_df)
    
    end_time = time.time()
    
    # Final Reporting
    total_graphs = len(metrics_df)
    total_nodes = metrics_df['node_count'].sum()
    total_edges = metrics_df['edge_count'].sum()
    
    avg_nodes = metrics_df['node_count'].mean()
    avg_edges = metrics_df['edge_count'].mean()
    avg_degree = metrics_df['average_degree'].mean()
    avg_isolated = metrics_df['isolated_nodes'].mean()
    avg_cc = metrics_df['connected_components'].mean()
    avg_cc_size = metrics_df['largest_cc_size'].mean()
    avg_cc_pct = metrics_df['largest_cc_pct'].mean()
    avg_density = metrics_df['graph_density'].mean()
    
    print("\n==========================================")
    print("PHASE 3 SUMMARY")
    print("==========================================")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Number of time windows: {total_graphs}")
    print(f"Window size: {window_size}s, Step: {step_size}s")
    print(f"Communication range: {comm_range}m")
    print(f"Total graph count: {total_graphs}")
    print(f"Total nodes across graphs: {total_nodes}")
    print(f"Total edges across graphs: {total_edges}")
    print(f"\nAverage nodes per graph: {avg_nodes:.2f}")
    print(f"Average edges per graph: {avg_edges:.2f}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Average isolated nodes: {avg_isolated:.2f}")
    print(f"Average connected components: {avg_cc:.2f}")
    print(f"Largest connected component (Avg Size): {avg_cc_size:.2f} ({avg_cc_pct:.2f}%)")
    print(f"Average graph density: {avg_density:.4f}")
    print("==========================================")
    print("\nPhase 3 complete.")

if __name__ == "__main__":
    run_graph_pipeline()
