import networkx as nx
import numpy as np
from networkx.algorithms.community.quality import modularity

def calculate_topology_metrics(G, labels, vids):
    """
    Calculate topology metrics for clusters on the graph.
    """
    label_dict = dict(zip(vids, labels))
    
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return {
            'total_edges': 0,
            'internal_edges': 0,
            'external_edges': 0,
            'internal_edge_ratio': 0.0,
            'clusters_with_disconnected_subgraphs': 0,
            'average_internal_edge_density': 0.0
        }
        
    internal_edges = 0
    external_edges = 0
    
    for u, v in G.edges():
        if label_dict[u] == label_dict[v]:
            internal_edges += 1
        else:
            external_edges += 1
            
    internal_edge_ratio = internal_edges / total_edges if total_edges > 0 else 0.0
    
    # Per-cluster connectivity
    clusters_with_disconnected_subgraphs = 0
    internal_densities = []
    
    unique_labels = set(labels)
    for label in unique_labels:
        nodes_in_cluster = [n for n in vids if label_dict[n] == label]
        subgraph = G.subgraph(nodes_in_cluster)
        
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        
        if n_nodes > 0:
            if not nx.is_connected(subgraph):
                clusters_with_disconnected_subgraphs += 1
                
        if n_nodes > 1:
            density = (2 * n_edges) / (n_nodes * (n_nodes - 1))
            internal_densities.append(density)
        else:
            internal_densities.append(0.0)
            
    avg_internal_edge_density = sum(internal_densities) / len(internal_densities) if internal_densities else 0.0
    
    return {
        'total_edges': total_edges,
        'internal_edges': internal_edges,
        'external_edges': external_edges,
        'internal_edge_ratio': internal_edge_ratio,
        'clusters_with_disconnected_subgraphs': clusters_with_disconnected_subgraphs,
        'average_internal_edge_density': avg_internal_edge_density
    }

def calculate_modularity(G, labels, vids):
    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        return np.nan
        
    label_dict = dict(zip(vids, labels))
    communities = {}
    for vid, lbl in label_dict.items():
        if lbl not in communities:
            communities[lbl] = set()
        communities[lbl].add(vid)
        
    communities_list = list(communities.values())
    
    if len(communities_list) <= 1:
        return 0.0
        
    try:
        return modularity(G, communities_list)
    except Exception:
        return np.nan

def calculate_conductance(G, cluster_nodes):
    if G.number_of_edges() == 0:
        return np.nan
    
    if len(cluster_nodes) == 0 or len(cluster_nodes) == G.number_of_nodes():
        return np.nan
        
    try:
        return nx.conductance(G, cluster_nodes)
    except Exception:
        return np.nan
