from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def calculate_clustering_metrics(X, labels):
    """
    Calculate feature-space clustering metrics.
    """
    n_labels = len(set(labels))
    
    if n_labels < 2 or n_labels == len(X):
        return {
            'silhouette_score': np.nan,
            'davies_bouldin_score': np.nan,
            'calinski_harabasz_score': np.nan
        }
        
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    
    return {
        'silhouette_score': sil,
        'davies_bouldin_score': db,
        'calinski_harabasz_score': ch
    }

def calculate_intra_cluster_distance(X, labels):
    """
    Calculate mean pairwise Euclidean distance within each cluster.
    Returns the average across all clusters.
    """
    unique_labels = set(labels)
    cluster_distances = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_points = len(cluster_points)
        
        if n_points <= 1:
            cluster_distances.append(0.0)
        else:
            diffs = cluster_points[:, np.newaxis, :] - cluster_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=-1))
            avg_dist = np.sum(dists) / (n_points * (n_points - 1))
            cluster_distances.append(avg_dist)
            
    if not cluster_distances:
        return np.nan
        
    return np.mean(cluster_distances)
