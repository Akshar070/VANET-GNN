# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from src.models.kmeans_baseline import run_kmeans, scale_features

def test_simple_separated_groups():
    X = np.array([
        [0, 0, 10], [0.1, 0.1, 10], [0, 0.1, 10],
        [100, 100, 30], [101, 100, 30], [100, 101, 30]
    ])
    labels, _ = run_kmeans(X, k=2, random_state=42)
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]

def test_reproducibility():
    np.random.seed(42)
    X = np.random.rand(20, 3)
    l1, i1 = run_kmeans(X, k=3, random_state=42)
    l2, i2 = run_kmeans(X, k=3, random_state=42)
    np.testing.assert_array_equal(l1, l2)
    assert i1 == i2

def test_invalid_nan():
    X = np.array([[0, 0, 10], [np.nan, 0, 10]])
    with pytest.raises(ValueError, match="NaN or inf"):
        scale_features(X)

def test_single_node_graph():
    X = np.array([[0, 0, 10]])
    labels, _ = run_kmeans(X, k=5)
    assert len(labels) == 1
    assert labels[0] == 0

def test_fewer_nodes_than_k():
    X = np.array([[0, 0, 10], [1, 1, 10]])
    labels, _ = run_kmeans(X, k=5)
    assert len(set(labels)) == 2

def test_identical_features():
    X = np.array([[0, 0, 10], [0, 0, 10], [0, 0, 10]])
    labels, _ = run_kmeans(X, k=2)
    assert len(set(labels)) == 1
