from types import SimpleNamespace

import networkx as nx
import torch
from torch_geometric.utils import to_undirected

from explainability import VIZ_MAX_NEIGHBOURS, get_viz_subgraph


def star(n_leaves):
    accounts = ["hub"] + [f"a{i}" for i in range(n_leaves)]
    G = nx.MultiDiGraph()
    for i in range(n_leaves):
        for _ in range(1 + (i % 3)):  # some neighbours trade more often
            G.add_edge("hub", f"a{i}")
    src = torch.zeros(n_leaves, dtype=torch.long)
    dst = torch.arange(1, n_leaves + 1)
    data = SimpleNamespace(accounts=accounts, edge_index=to_undirected(torch.stack([src, dst])))
    return data, G


def test_small_account_not_truncated():
    data, G = star(10)
    G_sub, hops, truncated = get_viz_subgraph("hub", data, G)
    assert not truncated and len(G_sub) == 11


def test_large_account_keeps_busiest_neighbours():
    data, G = star(500)
    G_sub, hops, truncated = get_viz_subgraph("hub", data, G)
    assert truncated and hops == 1
    assert len(G_sub) == VIZ_MAX_NEIGHBOURS + 1
    # every kept neighbour should have the max of 3 transactions with the hub
    kept = [n for n in G_sub if n != "hub"]
    assert all(G.number_of_edges("hub", n) == 3 for n in kept)
