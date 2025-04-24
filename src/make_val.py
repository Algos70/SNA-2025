import pandas as pd
import numpy as np

# 1) Load your full remapped edges
edges = pd.read_csv('data/edges_train_A_mapped.csv',
                    names=['src','dst','etype','ts'])

# 2) Pick a cut-off (e.g. last 10%)
t_cut = np.quantile(edges.ts, 0.9)
train_edges = edges[edges.ts <= t_cut]
val_edges   = edges[edges.ts >  t_cut]

# 3) Positive samples
val_pos = val_edges.copy()
val_pos['t0']   = val_pos.ts - 1
val_pos['t1']   = val_pos.ts
val_pos['label']= 1

# 4) Negative samples
num_neg    = len(val_pos)
num_nodes  = edges.src.max() + 1
neg_src    = np.random.randint(0, num_nodes, num_neg)
neg_dst    = np.random.randint(0, num_nodes, num_neg)
neg_etype  = np.random.choice(edges.etype.unique(), num_neg)
neg_t0     = val_pos['t0'].values
neg_t1     = val_pos['t1'].values

val_neg = pd.DataFrame({
    'src':   neg_src,
    'dst':   neg_dst,
    'etype': neg_etype,
    't0':    neg_t0,
    't1':    neg_t1,
    'label': np.zeros(num_neg, dtype=int)
})

# 5) Combine, shuffle, and write
val_queries = pd.concat([val_pos[['src','dst','etype','t0','t1','label']], val_neg])
val_queries = val_queries.sample(frac=1, random_state=42)
val_queries.to_csv('data/val_queries.csv', index=False, header=False)
