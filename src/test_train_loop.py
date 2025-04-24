import torch
import torch.nn as nn
from tqdm import tqdm
from model import GCNTemporalPredictor
from dataset_loader import load_dynamic_graph  # from step 3
import numpy as np
import random

def generate_positive_negative_edges(edge_index, num_nodes, num_samples=512):
    edge_set = set(map(tuple, edge_index.T.tolist()))
    pos_edges = random.sample(edge_set, min(num_samples, len(edge_set)))

    neg_edges = set()
    while len(neg_edges) < len(pos_edges):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        if (src, dst) not in edge_set:
            neg_edges.add((src, dst))

    return pos_edges, list(neg_edges)


def compute_edge_probs(node_embeds, edge_pairs):
    # Convert edge pairs to tensors
    src, dst = zip(*edge_pairs)
    src = torch.tensor(src, dtype=torch.long, device=node_embeds.device)
    dst = torch.tensor(dst, dtype=torch.long, device=node_embeds.device)

    src_vecs = node_embeds[src]
    dst_vecs = node_embeds[dst]

    scores = (src_vecs * dst_vecs).sum(dim=1)
    return scores

def test_single_step():
    dataset = load_dynamic_graph()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    num_node_features = dataset.features[0].shape[1]
    num_nodes = dataset.features[0].shape[0]
    model = GCNTemporalPredictor(node_features=num_node_features, hidden_dim=64, output_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    # Single time step
    t = 0
    x_t = torch.FloatTensor(dataset.features[t]).to(device)
    edge_index_t = torch.LongTensor(dataset.edge_indices[t]).to(device)
    edge_weight_t = torch.FloatTensor(dataset.edge_weights[t]).to(device)

    node_embeds = model(x_t, edge_index_t, edge_weight_t)

    future_edges = dataset.edge_indices[t + 1]
    pos_edges, neg_edges = generate_positive_negative_edges(future_edges, num_nodes, num_samples=128)

    edge_pairs = pos_edges + neg_edges
    labels = torch.cat([
        torch.ones(len(pos_edges)),
        torch.zeros(len(neg_edges))
    ]).to(device)

    scores = compute_edge_probs(node_embeds, edge_pairs).to(device)

    loss = loss_fn(scores, labels)
    print("✅ Sanity check — single step loss:", loss.item())

if __name__ == "__main__":
    test_single_step()
