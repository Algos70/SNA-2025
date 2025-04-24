import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

def load_dynamic_graph():
    edge_df = pd.read_csv("data/processed/edges_sampled.csv")
    node_feat_df = pd.read_csv("data/processed/node_features_encoded.csv")

    # Build a mapping of timestamp to edges
    timestamps = sorted(edge_df['timestamp'].unique())
    snapshot_dict = {ts: [] for ts in timestamps}

    for _, row in edge_df.iterrows():
        snapshot_dict[row['timestamp']].append((row['src_id'], row['dst_id']))

    edge_indices = []
    edge_weights = []

    for ts in timestamps:
        edges = np.array(snapshot_dict[ts]).T  # shape (2, num_edges)
        edge_indices.append(edges)
        edge_weights.append(np.ones(edges.shape[1]))  # dummy weights

    # Constant node features
    node_features = node_feat_df.iloc[:, 1:].values.astype(float)
    features = [node_features for _ in timestamps]

    # Dummy targets (not used in link prediction, just to satisfy API)
    dummy_targets = [np.zeros(node_features.shape[0]) for _ in timestamps]

    return DynamicGraphTemporalSignal(edge_indices, edge_weights, features, dummy_targets)
