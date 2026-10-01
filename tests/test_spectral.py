# pyrefly: ignore [missing-import]
import pytest
import numpy as np
import networkx as nx
from src.models.spectral_baseline import run_spectral_clustering, validate_adjacency

def test_simple_graph_communities():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    G.add_edges_from([(4, 5), (5, 6), (4, 6)])
    G.add_edge(3, 4)
    
    A = nx.to_numpy_array(G, nodelist=[1,2,3,4,5,6])
    labels = run_spectral_clustering(A, n_clusters=2, random_state=42)
    
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]

def test_two_node_graph():
    A = np.array([[0, 1], [1, 0]])
    labels = run_spectral_clustering(A, n_clusters=2)
    assert len(set(labels)) == 2

def test_no_edges():
    A = np.array([[0, 0], [0, 0]])
    labels = run_spectral_clustering(A, n_clusters=2)
    assert len(labels) == 2

def test_disconnected_graph():
    A = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    labels = run_spectral_clustering(A, n_clusters=2, random_state=42)
    assert len(labels) == 4
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]

def test_self_loops():
    A = np.array([[1, 1], [1, 0]])
    with pytest.raises(ValueError, match="self-loops"):
        validate_adjacency(A)

def test_asymmetric_matrix():
    A = np.array([[0, 1], [0, 0]])
    with pytest.raises(ValueError, match="symmetric"):
        validate_adjacency(A)

def test_nan_adjacency():
    A = np.array([[0, np.nan], [np.nan, 0]])
    with pytest.raises(ValueError, match="NaN"):
        validate_adjacency(A)

def test_reproducibility():
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    l1 = run_spectral_clustering(A, n_clusters=2, random_state=42)
    l2 = run_spectral_clustering(A, n_clusters=2, random_state=42)
    np.testing.assert_array_equal(l1, l2)

def test_k_larger_than_nodes():
    A = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="cannot be greater"):
        run_spectral_clustering(A, n_clusters=5)
