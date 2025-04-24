import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

DATA_PATH = "data"

def load_data():
    edges = pd.read_csv(
        os.path.join(DATA_PATH, "edges_train_A.csv"),
        header=None,
        names=["src_id", "dst_id", "edge_type", "timestamp"]
    )
    node_feats = pd.read_csv(os.path.join(DATA_PATH, "node_features.csv"))
    edge_type_feats = pd.read_csv(os.path.join(DATA_PATH, "edge_type_features.csv"))
    return edges, node_feats, edge_type_feats

def encode_categorical(df, skip_first=True):
    encoders = {}
    for col in df.columns[1 if skip_first else 0:]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # include -1 as string
        encoders[col] = le
    return df, encoders

def filter_and_sample_edges(edges, limit=4000):
    return edges.sort_values("timestamp", ascending=False).head(4000).sort_values("timestamp")


def preprocess_all():
    print("Loading raw data...")
    edges, node_feats, edge_type_feats = load_data()

    print("Encoding node features...")
    node_feats, node_encoders = encode_categorical(node_feats)

    node_id_map = {raw_id: idx for idx, raw_id in enumerate(node_feats.iloc[:, 0].values)}
    edges['src_id'] = edges['src_id'].map(node_id_map)
    edges['dst_id'] = edges['dst_id'].map(node_id_map)
    # Drop any edges that mapped to NaN (i.e., nodes not in node_feats)
    edges = edges.dropna().astype({'src_id': int, 'dst_id': int})

    print("Encoding edge type features...")
    edge_type_feats, edge_type_encoders = encode_categorical(edge_type_feats)

    print("Sampling temporal edges...")
    edges = filter_and_sample_edges(edges)

    print("Saving processed data...")
    os.makedirs("data/processed", exist_ok=True)
    edges.to_csv("data/processed/edges_sampled.csv", index=False)
    node_feats.to_csv("data/processed/node_features_encoded.csv", index=False)
    edge_type_feats.to_csv("data/processed/edge_type_features_encoded.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    preprocess_all()
