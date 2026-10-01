import numpy as np
import networkx as nx

def select_cluster_heads(G, clusters, prev_heads, prev_clusters, weights):
    """
    Selects cluster heads based on Degree, Centrality, Stability, and Speed.
    G: nx.Graph containing all vehicles and their attributes (speed).
    clusters: dict {cluster_id: set_of_vehicle_ids}
    prev_heads: dict {cluster_id: head_vehicle_id}
    prev_clusters: dict {cluster_id: set_of_vehicle_ids}
    weights: dict containing weights for the factors.
    
    Returns:
        new_heads: dict {cluster_id: head_vehicle_id}
        head_scores: dict {head_vehicle_id: score}
    """
    w_d = weights.get('degree_weight', 0.3)
    w_c = weights.get('centrality_weight', 0.25)
    w_s = weights.get('stability_weight', 0.3)
    w_v = weights.get('speed_penalty_weight', 0.15)
    
    new_heads = {}
    head_scores = {}
    
    for cid, members_set in clusters.items():
        members = list(members_set)
        if not members:
            continue
            
        if len(members) == 1:
            new_heads[cid] = members[0]
            head_scores[members[0]] = 1.0
            continue
            
        sub_G = G.subgraph(members)
        
        degrees = {v: G.degree(v) for v in members}  # Degree in full graph
        max_deg = max(degrees.values()) if degrees else 1
        min_deg = min(degrees.values()) if degrees else 0
        denom_deg = (max_deg - min_deg) if max_deg > min_deg else 1
            
        speeds = {v: G.nodes[v]['speed'] for v in members}
        max_speed = max(speeds.values()) if speeds else 1
        min_speed = min(speeds.values()) if speeds else 0
        denom_spd = (max_speed - min_speed) if max_speed > min_speed else 1
            
        # Closeness centrality inside the cluster subgraph
        centrality = nx.closeness_centrality(sub_G)
        max_cent = max(centrality.values()) if centrality else 1
        min_cent = min(centrality.values()) if centrality else 0
        denom_cent = (max_cent - min_cent) if max_cent > min_cent else 1
            
        scores = {}
        for v in members:
            deg_score = (degrees[v] - min_deg) / denom_deg
            cent_score = (centrality[v] - min_cent) / denom_cent
            spd_score = (speeds[v] - min_speed) / denom_spd
            
            stab_score = 0.0
            if cid in prev_heads and prev_heads[cid] == v:
                stab_score = 1.0
            elif cid in prev_clusters and v in prev_clusters[cid]:
                stab_score = 0.5
                
            score = (w_d * deg_score) + (w_c * cent_score) + (w_s * stab_score) - (w_v * spd_score)
            scores[v] = score
            
        best_head = max(scores.items(), key=lambda x: x[1])[0]
        new_heads[cid] = best_head
        head_scores[best_head] = scores[best_head]
        
    return new_heads, head_scores
