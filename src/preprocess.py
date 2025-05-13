# preprocess.py
import pandas as pd
import numpy as np
import json

def build_id_maps(node_feat_path: str,
                  edge_type_feat_path: str):
    """
    Read raw node and edge-type CSVs, build sorted maps,
    and dump them to JSON for reuse in inference.
    """
    # ---- Node IDs ----
    node_df = pd.read_csv(node_feat_path, header=None)
    unique_node_ids = sorted(int(x) for x in node_df.iloc[:,0].unique())
    node_map = {raw: new for new, raw in enumerate(unique_node_ids)}

    # ---- Edge-type IDs ----
    et_df = pd.read_csv(edge_type_feat_path, header=None)
    unique_etypes = sorted(int(x) for x in et_df.iloc[:,0].unique())
    etype_map = {raw: new for new, raw in enumerate(unique_etypes)}

    # ---- Persist maps ----
    with open("node_id_map.json", "w") as f:
        json.dump(node_map, f)
    with open("etype_id_map.json", "w") as f:
        json.dump(etype_map, f)

    print(f"Built maps for {len(node_map)} nodes and {len(etype_map)} edge-types")
    return node_map, etype_map


def remap_and_save_static(path_in: str,
                          path_out: str,
                          id_map: dict):
    """
    Remap the first column (ID) in path_in via id_map,
    drop that column, sort by new ID, and write only features.
    """
    df = pd.read_csv(path_in, header=None)
    # assume col0 is ID, the rest are features
    n_feats = df.shape[1] - 1
    df.columns = ["id"] + [f"feat_{i}" for i in range(1, n_feats+1)]

    # map IDs, drop unknown
    df["id"] = df["id"].map(id_map)
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    # reorder rows by mapped ID, then drop the ID column
    df = df.set_index("id").sort_index()
    feats = df.reset_index(drop=True)

    feats.to_csv(path_out, index=False, header=False)
    print(f"Saved {feats.shape[0]}×{feats.shape[1]} feature matrix to {path_out}")


def remap_edges_stream(in_path: str,
                       out_path: str,
                       node_map: dict,
                       etype_map: dict,
                       chunksize: int      = 1_000_000,
                       sample_frac: float  = None,
                       sample_n: int       = None,
                       stratify_by: str    = None,
                       n_bins: int         = 10,
                       seed: int           = 42):
    """
    Stream raw edges, remap src/dst/etype IDs, drop unknowns,
    and either write all or a sampled subset.

    - sample_frac (0–1) or sample_n (int) may be set, but not both.
    - stratify_by: 'etype' or 'ts' to sample per-group.
    """
    if (sample_frac is None) == (sample_n is None):
        # both None or both set is fine if you want ALL data (both None),
        # but if you specify one, make sure not the other.
        if sample_frac is not None and sample_n is not None:
            raise ValueError("Specify only one of sample_frac OR sample_n.")

    reader = pd.read_csv(
        in_path,
        header=None,
        names=["src","dst","etype","ts"],
        chunksize=chunksize,
    )
    first = True
    rng = np.random.default_rng(seed)

    for chunk in reader:
        # remap
        chunk["src"]   = chunk["src"].map(node_map)
        chunk["dst"]   = chunk["dst"].map(node_map)
        chunk["etype"] = chunk["etype"].map(etype_map)
        chunk = chunk.dropna(subset=["src","dst","etype"])
        chunk[["src","dst","etype"]] = chunk[["src","dst","etype"]].astype(int)

        # optionally bin ts for stratification
        if stratify_by == "ts":
            chunk["ts_bin"] = pd.qcut(chunk["ts"], q=n_bins, duplicates="drop")

        # sampling
        if sample_frac is None and sample_n is None:
            sampled = chunk  # no sampling, keep all
        elif stratify_by:
            sampled_parts = []
            group_field = "ts_bin" if stratify_by == "ts" else stratify_by
            for _, grp in chunk.groupby(group_field, group_keys=False):
                if sample_frac is not None:
                    k = max(1, int(round(len(grp) * sample_frac)))
                else:
                    k = min(len(grp), sample_n)
                idx = rng.choice(len(grp), size=k, replace=False)
                sampled_parts.append(grp.iloc[idx])
            sampled = pd.concat(sampled_parts, ignore_index=True)
        else:
            # uniform sampling
            total = len(chunk)
            if sample_frac is not None:
                k = max(1, int(round(total * sample_frac)))
            else:
                k = min(total, sample_n)
            idx = rng.choice(total, size=k, replace=False)
            sampled = chunk.iloc[idx]

        # drop helper column if present
        if "ts_bin" in sampled.columns:
            sampled = sampled.drop(columns=["ts_bin"])

        # write out
        sampled.to_csv(
            out_path,
            mode="w" if first else "a",
            index=False,
            header=False
        )
        first = False

    print(f"Saved remapped edges to {out_path}")


if __name__ == "__main__":
    # adjust these paths
    NODE_FEAT_IN   = "./data/node_features.csv"
    ETYPE_FEAT_IN  = "./data/edge_type_features.csv"
    NODE_FEAT_OUT  = "./data/node_features_mapped.csv"
    ETYPE_FEAT_OUT = "./data/edge_type_features_mapped.csv"
    EDGES_IN       = "./data/edges_train_A.csv"
    EDGES_OUT      = "./data/edges_train_A_mapped.csv"

    # pick your sampling strategy here:
    SAMPLE_FRAC    = 0.1        # e.g. 10% of edges
    SAMPLE_N       = None       # or set to an integer
    STRATIFY_BY    = "etype"    # or "ts", or None

    node_map, etype_map = build_id_maps(NODE_FEAT_IN, ETYPE_FEAT_IN)

    remap_and_save_static(NODE_FEAT_IN,  NODE_FEAT_OUT,  node_map)
    remap_and_save_static(ETYPE_FEAT_IN, ETYPE_FEAT_OUT, etype_map)

    remap_edges_stream(
        in_path      = EDGES_IN,
        out_path     = EDGES_OUT,
        node_map     = node_map,
        etype_map    = etype_map,
        chunksize    = 1_000_000,
        sample_frac  = SAMPLE_FRAC,
        sample_n     = SAMPLE_N,
        stratify_by  = STRATIFY_BY,
        n_bins       = 10,
        seed         = 42
    )

    print("Preprocessing complete.")
