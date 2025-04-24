# src/train.py
import torch
from torch.utils.data import DataLoader
from dataset import TemporalLinkPredictionDataset
from model import GConvGRULinkPredictor
from tqdm import tqdm
import argparse

def train(args):
    ds = TemporalLinkPredictionDataset(
        edges_path=args.edges,
        num_nodes=args.num_nodes,
        node_feat_path=args.node_feats,
        etype_feat_path=args.etype_feats,
        neg_sample_ratio=1.0
    )
    loader = DataLoader(ds, batch_size=None, shuffle=False)

    device = (
    torch.device('mps')
    if torch.backends.mps.is_available()
    else torch.device('cpu')
)   
    model = GConvGRULinkPredictor(
        in_channels=ds.node_features.shape[1],
        hidden_channels=args.hidden
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(1, args.epochs+1):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            x   = batch['x'].to(device)
            ei  = batch['edge_index'].to(device)
            pairs  = batch['pairs'].to(device)
            labels = batch['labels'].to(device)

            h = model(x, ei)
            preds = model.predict(h, pairs)
            loss = criterion(preds, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} avg loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.save_path)
    print("Training complete and model saved.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--edges',       default='data/edges_train_A_mapped.csv')
    p.add_argument('--node_feats',  default='data/node_features_mapped.csv')
    p.add_argument('--etype_feats', default='data/edge_type_features_mapped.csv')
    p.add_argument('--num_nodes',   type=int, required=True)
    p.add_argument('--hidden',      type=int, default=64)
    p.add_argument('--epochs',      type=int, default=10)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--save_path',   default='model.pth')
    args = p.parse_args()
    train(args)
