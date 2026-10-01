import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import scipy.sparse as sp

def validate_adjacency(A):
    if np.isnan(A).any() or np.isinf(A).any():
        raise ValueError("Adjacency matrix contains NaN or Inf.")
    
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("Adjacency matrix is not symmetric.")
        
    if np.any(np.diag(A) != 0):
        raise ValueError("Adjacency matrix contains self-loops.")

def run_spectral_clustering(A, n_clusters, random_state=42, n_init=20, assign_labels='kmeans'):
    """
    Run Spectral Clustering on adjacency matrix A.
    """
    validate_adjacency(A)
    
    n_nodes = A.shape[0]
    
    if n_clusters > n_nodes:
        raise ValueError(f"n_clusters ({n_clusters}) cannot be greater than number of nodes ({n_nodes}).")
        
    if n_clusters <= 1 or n_nodes <= 1:
        return np.zeros(n_nodes, dtype=int)
        
    if np.all(A == 0):
        # Disconnected graph with no edges
        # SpectralClustering with precomputed affinity all 0s will fail or produce arbitrary results.
        # Fall back safely
        return np.zeros(n_nodes, dtype=int)
        
    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed',
        random_state=random_state,
        n_init=n_init,
        assign_labels=assign_labels
    )
    labels = sc.fit_predict(A)
    return labels

def calculate_eigenvalues(A):
    """
    Calculate the eigenvalues of the normalized Laplacian.
    L_sym = I - D^(-1/2) A D^(-1/2)
    """
    validate_adjacency(A)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    degrees = np.sum(A, axis=1)
    
    d_inv_sqrt = np.zeros_like(degrees, dtype=float)
    nonzero_mask = degrees > 0
    d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
    
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    I = np.eye(n)
    L_sym = I - D_inv_sqrt @ A @ D_inv_sqrt
    
    eigenvalues = np.linalg.eigvalsh(L_sym)
    # Ensure numerical stability (sometimes very small negative eigs exist)
    eigenvalues[eigenvalues < 0] = 0.0
    return np.sort(eigenvalues)

def construct_adjacency_from_graph(G, vids):
    """
    Constructs adjacency matrix A for the exact ordering of vids.
    """
    A = nx.to_numpy_array(G, nodelist=vids, weight=None) 
    np.fill_diagonal(A, 0)
    return A
