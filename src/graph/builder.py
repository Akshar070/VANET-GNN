import os
import networkx as nx
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import cKDTree
from pathlib import Path

def build_window_graph(snapshot_df, comm_range):
    """
    Builds a NetworkX graph for a single window snapshot using KDTree.
    """
    G = nx.Graph()
    if snapshot_df.empty:
        return G
        
    coords = snapshot_df[['x', 'y']].values
    vids = snapshot_df['vehicle_id'].values
    speeds = snapshot_df['speed'].values
    times = snapshot_df['time'].values
    
    for i, vid in enumerate(vids):
        G.add_node(vid, x=coords[i, 0], y=coords[i, 1], speed=speeds[i], time=times[i])
        
    if len(coords) < 2:
        return G
        
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=comm_range)
    
    for i, j in pairs:
        vid_i = vids[i]
        vid_j = vids[j]
        dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
        G.add_edge(vid_i, vid_j, distance=dist, weight=1.0)
        
    return G

def calculate_graph_metrics(G, window_id):
    """
    Calculates statistics for a given graph.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes > 0:
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / num_nodes
        min_degree = min(degrees)
        max_degree = max(degrees)
        isolated_nodes = sum(1 for d in degrees if d == 0)
    else:
        avg_degree = 0.0
        min_degree = 0
        max_degree = 0
        isolated_nodes = 0
        
    if num_nodes > 1:
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    else:
        density = 0.0
        
    num_components = nx.number_connected_components(G) if num_nodes > 0 else 0
    if num_components > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_size = len(largest_cc)
        largest_cc_pct = (largest_cc_size / num_nodes) * 100
    else:
        largest_cc_size = 0
        largest_cc_pct = 0.0
        
    return {
        'window_id': window_id,
        'node_count': num_nodes,
        'edge_count': num_edges,
        'average_degree': avg_degree,
        'minimum_degree': min_degree,
        'maximum_degree': max_degree,
        'isolated_nodes': isolated_nodes,
        'connected_components': num_components,
        'largest_cc_size': largest_cc_size,
        'largest_cc_pct': largest_cc_pct,
        'graph_density': density
    }

def construct_dynamic_graphs(snapshots_df, metadata_df, output_dir, comm_range):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graphs = {}
    graph_metrics = []
    validation_failures = 0
    
    print(f"Building graphs for {len(metadata_df)} windows with range {comm_range}m ...")
    
    for _, row in metadata_df.iterrows():
        wid = int(row['window_id'])
        window_snapshots = snapshots_df[snapshots_df['window_id'] == wid]
        
        G = build_window_graph(window_snapshots, comm_range)
        
        has_self_loops = nx.number_of_selfloops(G) > 0
        has_invalid_distance = False
        for u, v, d in G.edges(data=True):
            if d['distance'] > comm_range:
                has_invalid_distance = True
                break
                
        if has_self_loops or has_invalid_distance:
            print(f"Validation failed for window {wid}! Self loops: {has_self_loops}, Invalid distance: {has_invalid_distance}")
            validation_failures += 1
            
        metrics = calculate_graph_metrics(G, wid)
        metrics['start_time'] = row['start_time']
        metrics['end_time'] = row['end_time']
        
        graph_metrics.append(metrics)
        
        graph_path = output_dir / f"graph_{wid:04d}.gpickle"
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
            
    metrics_df = pd.DataFrame(graph_metrics)
    metrics_df = metrics_df[['window_id', 'start_time', 'end_time', 'node_count', 'edge_count', 'average_degree', 
                             'minimum_degree', 'maximum_degree', 'isolated_nodes', 'connected_components', 
                             'largest_cc_size', 'largest_cc_pct', 'graph_density']]
                             
    metrics_df.to_csv(output_dir / "graph_metadata.csv", index=False)
    print(f"Saved graphs and metadata to {output_dir}")
    print(f"Validation failures: {validation_failures}")
    
    return metrics_df
