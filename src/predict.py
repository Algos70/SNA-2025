import torch
import pandas as pd
import numpy as np
from model import GCNTemporalPredictor
from dataset_loader import load_dynamic_graph

def load_model(model_path, num_node_features, device):
    model = GCNTemporalPredictor(node_features=num_node_features, hidden_dim=64, output_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_link_prob(src_id, dst_id, node_embeds):
    # src/dst must be within the valid node ID range
    try:
        src_vec = node_embeds[src_id]
        dst_vec = node_embeds[dst_id]
        score = torch.dot(src_vec, dst_vec).item()
        return torch.sigmoid(torch.tensor(score)).item()
    except IndexError:
        return 0.0  # default score for unknown/missing nodes

def predict_all():
    # Load graph and model
    dataset = load_dynamic_graph()
    device = torch.device('cpu')
    num_node_features = dataset.features[0].shape[1]
    model = load_model("output/gcn_temporal.pt", num_node_features, device)

    # Load test queries
    test_df = pd.read_csv(
    "data/input_A.csv",
    header=None,
    names=["src_id", "dst_id", "edge_type", "start_time", "end_time"]
)
    test_df['score'] = 0.0

    # Load node ID remapping
    node_feat_df = pd.read_csv("data/processed/node_features_encoded.csv")
    node_id_map = {raw_id: idx for idx, raw_id in enumerate(node_feat_df.iloc[:, 0].values)}

    # Remap src_id and dst_id in the test set
    test_df["src_id"] = test_df["src_id"].map(node_id_map)
    test_df["dst_id"] = test_df["dst_id"].map(node_id_map)


    # Drop test queries with unknown nodes
    test_df = test_df.dropna().astype({"src_id": int, "dst_id": int})
    max_train_ts = 1413867600
    test_df = test_df[test_df["start_time"] <= max_train_ts]
    print("Example remapped src/dst IDs:")
    print(test_df[["src_id", "dst_id"]].head())
    print("Max src_id index:", test_df["src_id"].max())
    print("Max dst_id index:", test_df["dst_id"].max())


    # We'll embed all snapshots up to the latest timestamp
    edge_index_all = dataset.edge_indices
    edge_weight_all = dataset.edge_weights
    features_all = dataset.features
    timestamps = list(range(len(edge_index_all)))

    # Map: timestamp → index in temporal sequence
    # We assume timestamps in edge set are already sorted
    edge_df = pd.read_csv("data/processed/edges_sampled.csv")
    unique_ts = sorted(edge_df['timestamp'].unique())
    ts_to_idx = {ts: idx for idx, ts in enumerate(unique_ts)}

    # For each test query:
    for i, row in test_df.iterrows():
        src_id = row["src_id"]
        dst_id = row["dst_id"]
        start_time = row["start_time"]


        # Find the closest snapshot before start_time
        past_ts = [ts for ts in unique_ts if ts <= start_time]
        if not past_ts:
            test_df.at[i, 'score'] = 0.0
            continue

        closest_ts = past_ts[-1]
        t_idx = ts_to_idx[closest_ts]
        print(f"Closest timestamp for start_time={start_time}: {closest_ts} → snapshot index {t_idx}")


        x_t = torch.FloatTensor(features_all[t_idx]).to(device)
        edge_index_t = torch.LongTensor(edge_index_all[t_idx]).to(device)
        edge_weight_t = torch.FloatTensor(edge_weight_all[t_idx]).to(device)

        with torch.no_grad():
            node_embeds = model(x_t, edge_index_t, edge_weight_t)
            print("Embedding sample (first row):", node_embeds[0][:5])


        prob = predict_link_prob(src_id, dst_id, node_embeds)
        if i < 5:
            print(f"[Sample] src={src_id}, dst={dst_id}, score={prob:.4f}")
        test_df.at[i, 'score'] = prob

    # Save output
    output_df = test_df[["src_id", "dst_id", "edge_type", "start_time", "end_time", "score"]]
    output_df.to_csv("output/output_A.csv", index=False)
    print("✅ Predictions written to output/output_A.csv")

if __name__ == "__main__":
    predict_all()
