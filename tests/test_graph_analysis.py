import networkx as nx

from graph_analysis import canonicalize_cycle, find_long_cycles


def test_long_cycles_match_networkx():
    G = nx.gnp_random_graph(60, 0.08, directed=True, seed=3)
    G = nx.relabel_nodes(G, {i: f"n{i:03d}" for i in G})
    expected = {canonicalize_cycle(c) for c in nx.simple_cycles(G, length_bound=6) if len(c) >= 4}
    assert find_long_cycles(G, set(G)) == expected


def test_long_cycles_deterministic():
    G = nx.gnp_random_graph(80, 0.06, directed=True, seed=7)
    assert find_long_cycles(G, set(G)) == find_long_cycles(G, set(G))
