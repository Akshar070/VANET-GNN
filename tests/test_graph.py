import pytest
import pandas as pd
import numpy as np
import networkx as nx
from src.graph.builder import build_window_graph
from src.data.windowing import create_time_windows
import tempfile
from pathlib import Path

def test_distance_within_range():
    # Test 1 & 2: 50m and 200m apart (edges should exist if comm_range=200)
    df = pd.DataFrame({
        'vehicle_id': [1, 2, 3],
        'time': [0.0, 0.0, 0.0],
        'x': [0.0, 50.0, 200.0],
        'y': [0.0, 0.0, 0.0],
        'speed': [10.0, 10.0, 10.0]
    })
    
    G = build_window_graph(df, 200.0)
    
    assert G.has_edge(1, 2) # distance = 50 <= 200
    assert G.has_edge(1, 3) # distance = 200 <= 200
    
def test_distance_outside_range():
    # Test 3: 201m apart (no edge)
    df = pd.DataFrame({
        'vehicle_id': [1, 2],
        'time': [0.0, 0.0],
        'x': [0.0, 201.0],
        'y': [0.0, 0.0],
        'speed': [10.0, 10.0]
    })
    
    G = build_window_graph(df, 200.0)
    assert not G.has_edge(1, 2)

def test_no_self_loops():
    # Test 4: Vehicle compared with itself
    df = pd.DataFrame({
        'vehicle_id': [1],
        'time': [0.0],
        'x': [0.0],
        'y': [0.0],
        'speed': [10.0]
    })
    
    G = build_window_graph(df, 200.0)
    assert not G.has_edge(1, 1)

def test_identical_positions():
    # Test 5: Two vehicles at exact same location
    df = pd.DataFrame({
        'vehicle_id': [1, 2],
        'time': [0.0, 0.0],
        'x': [50.0, 50.0],
        'y': [50.0, 50.0],
        'speed': [10.0, 10.0]
    })
    
    G = build_window_graph(df, 200.0)
    assert G.has_edge(1, 2)
    assert G.edges[1, 2]['distance'] == 0.0

def test_empty_window():
    # Test 6: Empty window handled safely
    df = pd.DataFrame(columns=['vehicle_id', 'time', 'x', 'y', 'speed'])
    G = build_window_graph(df, 200.0)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0

def test_single_vehicle():
    # Test 7: Single vehicle has 1 node, 0 edges
    df = pd.DataFrame({
        'vehicle_id': [1],
        'time': [0.0],
        'x': [0.0],
        'y': [0.0],
        'speed': [10.0]
    })
    
    G = build_window_graph(df, 200.0)
    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0

def test_temporal_leakage():
    # Test 8: Multiple windows, vehicle state does not leak
    # We use the windowing logic for this
    df = pd.DataFrame({
        'vehicle_id': [1, 1, 1, 1],
        'time': [5.0, 15.0, 25.0, 35.0],
        'x': [10.0, 20.0, 30.0, 40.0],
        'y': [0.0, 0.0, 0.0, 0.0],
        'speed': [5.0, 5.0, 5.0, 5.0]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_df, snap_df = create_time_windows(df, window_size=10, step_size=10, output_dir=tmpdir)
        
        # Window 0: [0, 10), state should be x=10
        w0 = snap_df[snap_df['window_id'] == 0]
        assert w0.iloc[0]['x'] == 10.0
        
        # Window 1: [10, 20), state should be x=20
        w1 = snap_df[snap_df['window_id'] == 1]
        assert w1.iloc[0]['x'] == 20.0
        
        # Window 2: [20, 30), state should be x=30
        w2 = snap_df[snap_df['window_id'] == 2]
        assert w2.iloc[0]['x'] == 30.0
