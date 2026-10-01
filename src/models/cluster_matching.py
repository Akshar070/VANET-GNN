import numpy as np
from scipy.optimize import linear_sum_assignment

def match_clusters(prev_clusters, curr_clusters):
    """
    Matches clusters temporally using the Hungarian algorithm to maximize Jaccard similarity.
    prev_clusters: dict {cluster_id: set_of_vehicle_ids}
    curr_clusters: dict {local_cluster_id: set_of_vehicle_ids}
    
    Returns:
        mapping: dict {local_cluster_id: matched_global_cluster_id}
    """
    if not prev_clusters:
        return {cid: cid for cid in curr_clusters}
    
    prev_ids = list(prev_clusters.keys())
    curr_ids = list(curr_clusters.keys())
    
    cost_matrix = np.zeros((len(curr_ids), len(prev_ids)))
    
    for i, cid in enumerate(curr_ids):
        c_set = curr_clusters[cid]
        for j, pid in enumerate(prev_ids):
            p_set = prev_clusters[pid]
            intersection = len(c_set.intersection(p_set))
            union = len(c_set.union(p_set))
            jaccard = intersection / union if union > 0 else 0.0
            # Linear sum assignment minimizes cost, so we use negative Jaccard
            cost_matrix[i, j] = -jaccard
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    mapping = {}
    used_pids = set()
    
    for i, j in zip(row_ind, col_ind):
        # Only assign if there is some overlap (Jaccard > 0)
        if cost_matrix[i, j] < 0:
            mapping[curr_ids[i]] = prev_ids[j]
            used_pids.add(prev_ids[j])
            
    # For any unmatched curr_cluster, assign a new unique global ID
    max_id = max(prev_ids + curr_ids) if prev_ids else 0
    next_id = max_id + 1
    
    for cid in curr_ids:
        if cid not in mapping:
            while next_id in used_pids or next_id in prev_ids:
                next_id += 1
            mapping[cid] = next_id
            used_pids.add(next_id)
            next_id += 1
            
    return mapping
