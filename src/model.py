import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class GCNTemporalPredictor(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim):
        super().__init__()
        self.recurrent = GConvGRU(node_features, hidden_dim, K=2)  # GCN layers
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.recurrent(x, edge_index, edge_weight)
        h = self.linear(h)
        return F.normalize(h, p=2, dim=1)  # normalize node embeddings
