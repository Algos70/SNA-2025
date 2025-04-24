# src/model.py
import torch
from torch_geometric_temporal.nn.recurrent import GConvGRU

class GConvGRULinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.recurrent = GConvGRU(in_channels, hidden_channels, K=2)
        self.link_pred = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        h = self.recurrent(x, edge_index)
        return h

    def predict(self, h, pairs):
        # pairs: [batch_size, 2]
        src, dst = pairs[:,0], pairs[:,1]
        h_src = h[src]
        h_dst = h[dst]
        h_cat = torch.cat([h_src, h_dst], dim=1)
        return self.link_pred(h_cat).squeeze()
