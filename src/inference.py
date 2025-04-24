# src/inference.py
import torch
import pandas as pd
from model import GConvGRULinkPredictor
import argparse

def infer(args):
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else
                      'cpu')
    dummy = torch.load(args.model_path, map_location=device)
    # load node features first to infer in_channels
    nf_df = pd.read_csv(args.node_feats, header=None)
    in_ch = nf_df.shape[1]
    model = GConvGRULinkPredictor(in_ch, args.hidden).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # load full graph for final embedding
    edges = pd.read_csv(args.edges, names=['src','dst','etype','ts'])
    edge_index = torch.tensor([edges.src, edges.dst], dtype=torch.long).to(device)
    x = torch.tensor(
        pd.read_csv(args.node_feats, header=None).values,
        dtype=torch.float
    ).to(device)

    with torch.no_grad():
        h = model(x, edge_index)
        q = pd.read_csv(args.query, names=['src','dst','etype','t0','t1'])
        pairs = torch.tensor(q[['src','dst']].values, dtype=torch.long).to(device)
        preds = model.predict(h, pairs).detach().cpu().numpy()

    out = q.copy()
    out['score'] = preds
    out[['src','dst','etype','t0','t1','score']].to_csv(
        args.output, index=False, header=False
    )
    print(f"Wrote predictions to {args.output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='model/model.pth')
    p.add_argument('--edges',      default='data/edges_train_A_mapped.csv')
    p.add_argument('--node_feats', default='data/node_features_mapped.csv')
    p.add_argument('--query',      default='data/input_A.csv')
    p.add_argument('--output',     default='output/output_A.csv')
    p.add_argument('--hidden',     type=int, default=64)
    args = p.parse_args()
    infer(args)
