# src/dataset.py
import torch
import pandas as pd
import numpy as np

class TemporalLinkPredictionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 edges_path: str,
                 num_nodes: int,
                 node_feat_path: str,
                 etype_feat_path: str,
                 neg_sample_ratio: float = 1.0):
        """
        edges_path: remapped CSV with columns src,dst,etype,ts
        """
        self.edges = pd.read_csv(edges_path,
                                 names=['src','dst','etype','ts'])
        self.timestamps = sorted(self.edges['ts'].unique())
        self.node_features = torch.tensor(
            pd.read_csv(node_feat_path, header=None).values,
            dtype=torch.float
        )
        self.etype_features = torch.tensor(
            pd.read_csv(etype_feat_path, header=None).values,
            dtype=torch.float
        )
        self.num_nodes = num_nodes
        self.neg_ratio = neg_sample_ratio

    def __len__(self):
        # we predict edges in each adjacent time slot
        return len(self.timestamps) - 1

    def __getitem__(self, idx):
        t = self.timestamps[idx]
        t_next = self.timestamps[idx + 1]

        # graph up to time t
        hist = self.edges[self.edges.ts <= t]
        edge_index = torch.tensor(
            [hist.src.values, hist.dst.values],
            dtype=torch.long
        )

        # positives in (t, t_next]
        pos = self.edges[
            (self.edges.ts > t) &
            (self.edges.ts <= t_next)
        ][['src','dst']].values

        # negative sampling: uniform random pairs
        K = len(pos)
        neg_src = np.random.randint(0, self.num_nodes, size=int(K*self.neg_ratio))
        neg_dst = np.random.randint(0, self.num_nodes, size=int(K*self.neg_ratio))
        neg = np.stack([neg_src, neg_dst], axis=1)

        # assemble training examples
        pairs = torch.tensor(
            np.vstack([pos, neg]),
            dtype=torch.long
        )
        labels = torch.cat([
            torch.ones(len(pos)),
            torch.zeros(len(neg))
        ])

        return {
            'x': self.node_features,
            'edge_index': edge_index,
            'pairs': pairs,
            'labels': labels,
        }
