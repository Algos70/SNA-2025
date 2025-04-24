# preprocess.py
import pandas as pd
import json

def build_id_maps(node_feat_path, edge_type_feat_path):
    node_df = pd.read_csv(node_feat_path, header=None)
    node_ids = node_df.iloc[:,0].unique()
    node_map = {int(raw): int(new) for new, raw in enumerate(node_ids)}

    etype_df = pd.read_csv(edge_type_feat_path, header=None)
    etype_ids = etype_df.iloc[:,0].unique()
    etype_map = {int(raw): int(new) for new, raw in enumerate(etype_ids)}

    with open('node_id_map.json', 'w') as f:
        json.dump(node_map, f)
    with open('etype_id_map.json', 'w') as f:
        json.dump(etype_map, f)

    return node_map, etype_map

def remap_and_save_static(path_in, path_out, id_map):
    df = pd.read_csv(path_in, header=None)
    df.iloc[:,0] = df.iloc[:,0].map(id_map)
    df.to_csv(path_out, index=False, header=False)

def remap_edges_stream(in_path, out_path, node_map, etype_map, chunksize=1_000_000):
    reader = pd.read_csv(in_path, header=None,
                         names=['src','dst','etype','ts'],
                         chunksize=chunksize)
    first = True
    for chunk in reader:
        chunk['src']   = chunk['src'].map(node_map)
        chunk['dst']   = chunk['dst'].map(node_map)
        chunk['etype'] = chunk['etype'].map(etype_map)
        chunk.to_csv(out_path,
                     mode='w' if first else 'a',
                     index=False, header=False)
        first = False

if __name__ == '__main__':
    node_map, etype_map = build_id_maps(
        'data/node_features.csv',
        'data/edge_type_features.csv'
    )
    remap_and_save_static(
        'data/node_features.csv',
        'data/node_features_mapped.csv',
        node_map
    )
    remap_and_save_static(
        'data/edge_type_features.csv',
        'data/edge_type_features_mapped.csv',
        etype_map
    )
    remap_edges_stream(
        'data/edges_train_A.csv',
        'data/edges_train_A_mapped.csv',
        node_map,
        etype_map
    )
    print("Preprocessing done.")
