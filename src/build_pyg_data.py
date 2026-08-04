"""
Convert the transaction graph + account feature table into a PyTorch
Geometric Data object for node classification (GraphSAGE baseline).

Edge representation choice: the collapsed simple directed graph (1.02M
unique edges, same one built in graph_analysis.py/feature_engineering.py)
rather than the full MultiDiGraph (5.08M edges). Repeated transactions
between the same pair only duplicate connectivity, not new topology —
that repeat-count signal is already captured as node features
(transaction_count, total_sent, total_received, etc.). Using the full
multi-edge set would make every epoch ~5x more expensive for no new
structural information. The collapsed edges are then made undirected
(both directions added) since in/out-degree are already separate node
features, so directionality isn't lost, only supplemented for message
passing — standard practice for GNN fraud-detection baselines.

Node feature columns: the same 13 modeling features used in
train_baseline.py (account, community_id, is_illicit excluded/handled
separately).

Run from the project root with the venv activated:

    python src/build_pyg_data.py
"""

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_analysis as ga

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "account_features.csv"
PYG_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pyg_data.pt"

TARGET = "is_illicit"
EXCLUDED_COLS = ["account", "community_id", TARGET]


def build_data() -> Data:
    print("Loading feature table...")
    features_df = pd.read_csv(FEATURES_CSV)
    features_df["in_out_amount_ratio"] = features_df["in_out_amount_ratio"].fillna(-1)

    print("Loading graph and collapsing to simple DiGraph...")
    G = ga.load_graph()
    simple = ga.build_simple_digraph(G)

    graph_accounts = set(simple.nodes())
    feature_accounts = set(features_df["account"])
    assert graph_accounts == feature_accounts, (
        f"Node set mismatch between graph ({len(graph_accounts)}) and "
        f"feature table ({len(feature_accounts)}) — cannot safely align."
    )

    # Node order = feature table row order; this defines the node index space.
    accounts = features_df["account"].tolist()
    account_to_idx = {acct: i for i, acct in enumerate(accounts)}

    feature_cols = [c for c in features_df.columns if c not in EXCLUDED_COLS]
    print(f"  Node features ({len(feature_cols)}): {feature_cols}")
    x = torch.tensor(features_df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(features_df[TARGET].values, dtype=torch.long)

    print("Building edge_index from collapsed directed edges...")
    t0 = time.time()
    src = np.fromiter((account_to_idx[u] for u, v in simple.edges()), dtype=np.int64, count=simple.number_of_edges())
    dst = np.fromiter((account_to_idx[v] for u, v in simple.edges()), dtype=np.int64, count=simple.number_of_edges())
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=len(accounts))
    print(f"  Done in {time.time() - t0:.1f}s: {edge_index.shape[1]:,} directed edges after undirecting")

    data = Data(x=x, edge_index=edge_index, y=y)
    data.accounts = accounts
    data.feature_cols = feature_cols
    return data


def main() -> None:
    data = build_data()
    print()
    print("=== PyG Data object ===")
    print(data)
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges (directed, post-undirecting): {data.num_edges:,}")
    print(f"Node feature dim: {data.num_node_features}")
    print(f"Positive labels: {int(data.y.sum()):,} ({100*data.y.float().mean().item():.4f}%)")
    print(f"Isolated nodes: {(torch.bincount(data.edge_index[0], minlength=data.num_nodes) == 0).sum().item():,}")

    torch.save(data, PYG_DATA_PATH)
    print(f"\nSaved PyG Data object to {PYG_DATA_PATH}")


if __name__ == "__main__":
    main()
