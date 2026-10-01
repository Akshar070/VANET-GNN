"""Distance-based VANET graph for one time window."""

import numpy as np
import networkx as nx

def build_graph(vehicle_frame, communication_range_m=200):
    G = nx.Graph()

    ids = vehicle_frame["vehicle_id"].to_numpy()
    xy = vehicle_frame[["x", "y"]].to_numpy(dtype=float)

    for vehicle_id in ids:
        G.add_node(int(vehicle_id))

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            distance = np.linalg.norm(xy[i] - xy[j])
            if distance <= communication_range_m:
                G.add_edge(int(ids[i]), int(ids[j]))

    return G
