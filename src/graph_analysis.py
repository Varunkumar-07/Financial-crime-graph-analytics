"""
Graph analysis: centrality, Louvain communities and cycles (length 2-6).

- Betweenness is approximate (k=500 samples, ~3-4 min).
- 2/3-cycles are exact across the whole graph.
- 4-6 cycles are exact everywhere (rank-ordered depth-limited search).

Usage: python src/graph_analysis.py
"""

import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "transaction_graph.gpickle"
SUMMARY_MD = PROJECT_ROOT / "notebooks" / "graph_analysis_summary.md"
CLUSTER_PNG = PROJECT_ROOT / "notebooks" / "suspicious_cluster.png"

BETWEENNESS_K = 500


def load_graph() -> nx.MultiDiGraph:
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def build_simple_digraph(G: nx.MultiDiGraph) -> nx.DiGraph:
    simple = nx.DiGraph()
    simple.add_nodes_from(G.nodes())
    for u, v in G.edges():
        simple.add_edge(u, v)
    return simple


def build_undirected_weighted(simple: nx.DiGraph) -> nx.Graph:
    undirected = nx.Graph()
    undirected.add_nodes_from(simple.nodes())
    for u, v in simple.edges():
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += 1
        else:
            undirected.add_edge(u, v, weight=1)
    return undirected


def get_illicit_nodes_and_edges(G: nx.MultiDiGraph):
    illicit_nodes = set()
    illicit_edge_pairs = set()
    for u, v, is_laundering in G.edges(data="is_laundering"):
        if is_laundering == 1:
            illicit_nodes.add(u)
            illicit_nodes.add(v)
            illicit_edge_pairs.add((u, v))
    return illicit_nodes, illicit_edge_pairs


def canonicalize_cycle(nodes: list) -> tuple:
    min_idx = min(range(len(nodes)), key=lambda i: nodes[i])
    return tuple(nodes[min_idx:] + nodes[:min_idx])


def cycle_touches_illicit(cycle_nodes: tuple, illicit_edge_pairs: set) -> bool:
    n = len(cycle_nodes)
    for i in range(n):
        u, v = cycle_nodes[i], cycle_nodes[(i + 1) % n]
        if (u, v) in illicit_edge_pairs:
            return True
    return False


def find_2_and_3_cycles(simple: nx.DiGraph):
    edge_set = set(simple.edges())
    two = set()
    for u, v in edge_set:
        if u != v and (v, u) in edge_set:
            two.add(canonicalize_cycle([u, v]))

    three = set()
    for u, v in edge_set:
        if u == v:  # skip self-loops
            continue
        for w in simple.successors(v):
            if w != u and w != v and (w, u) in edge_set:
                three.add(canonicalize_cycle([u, v, w]))
    return two, three


def find_small_scc_cycles(simple: nx.DiGraph, small_sccs: list):
    found = set()
    for scc in small_sccs:
        found |= find_long_cycles(simple, scc)
    return found


def find_long_cycles(simple: nx.DiGraph, nodes: set) -> set:
    """Exact 4-6 length cycles within `nodes`.
    Each cycle is found once, from its smallest node, by a depth-limited
    DFS that only visits larger nodes. Deterministic and fast (~0.1s on
    the 17K-node giant SCC, where nx.simple_cycles never finished).
    """
    sub = simple.subgraph(nodes)
    order = sorted(sub.nodes())
    rank = {n: i for i, n in enumerate(order)}
    succ = {n: sorted(sub.successors(n), key=rank.__getitem__) for n in order}
    found = set()
    for start in order:
        r0 = rank[start]
        path, on_path, stack = [start], {start}, [iter(succ[start])]
        while stack:
            for w in stack[-1]:
                if w == start:
                    if len(path) >= 4:
                        found.add(tuple(path))  # already starts at its smallest node
                elif rank[w] > r0 and w not in on_path and len(path) < 6:
                    path.append(w)
                    on_path.add(w)
                    stack.append(iter(succ[w]))
                    break
            else:
                stack.pop()
                on_path.discard(path.pop())
    return found


CYCLES_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "all_cycles_cache.pkl"
METRICS_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "graph_metrics_cache.pkl"


def get_graph_metrics_cached(simple: nx.DiGraph, cache_path=METRICS_CACHE_PATH):
    """Approximate betweenness and Louvain communities, cached on disk.
    Both are seeded, so the cache matches a fresh computation.
    Returns (betweenness dict, list of community sets).
    """
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"Loaded cached betweenness/communities from {cache_path}")
        return cached["betweenness"], cached["communities"]

    print(f"Computing approximate betweenness centrality (k={BETWEENNESS_K})...")
    t0 = time.time()
    betweenness = nx.betweenness_centrality(simple, k=BETWEENNESS_K, seed=42, normalized=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Running Louvain community detection...")
    t0 = time.time()
    undirected = build_undirected_weighted(simple)
    communities = nx.community.louvain_communities(undirected, weight="weight", seed=42)
    print(f"  Done in {time.time() - t0:.1f}s, {len(communities)} communities")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"betweenness": betweenness, "communities": communities}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return betweenness, communities


def compute_cycle_analysis(simple: nx.DiGraph) -> dict:
    """Runs all cycle detection and returns the results, including the
    flat `all_cycles` set used by explainability.py and the API."""
    two_cycles, three_cycles = find_2_and_3_cycles(simple)

    sccs = list(nx.strongly_connected_components(simple))
    nontrivial_sccs = [s for s in sccs if len(s) > 1]
    giant_scc = max(nontrivial_sccs, key=len)
    small_sccs = [s for s in nontrivial_sccs if s is not giant_scc]

    small_scc_cycles = find_small_scc_cycles(simple, small_sccs)
    giant_scc_cycles = find_long_cycles(simple, giant_scc)

    all_exact_cycles = two_cycles | three_cycles | small_scc_cycles
    all_cycles = all_exact_cycles | giant_scc_cycles

    return {
        "two_cycles": two_cycles,
        "three_cycles": three_cycles,
        "small_scc_cycles": small_scc_cycles,
        "giant_scc_cycles": giant_scc_cycles,
        "all_exact_cycles": all_exact_cycles,
        "all_cycles": all_cycles,
        "nontrivial_scc_count": len(nontrivial_sccs),
        "giant_scc_size": len(giant_scc),
        "small_scc_count": len(small_sccs),
        "small_scc_max_size": max(len(s) for s in small_sccs),
    }


def get_cycle_analysis_cached(simple: nx.DiGraph, cache_path=CYCLES_CACHE_PATH) -> dict:
    """Cached compute_cycle_analysis() (detection takes ~10s).
    Old-format caches (including ones from the old time-capped search)
    are ignored and rebuilt.
    """
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and "all_cycles" in cached and not cached.get("hit_cap"):
            print(f"Loaded cached cycle analysis from {cache_path} "
                  f"({len(cached['all_cycles']):,} cycles)")
            return cached
        print(f"Cache at {cache_path} is in an old/incompatible format — recomputing.")

    result = compute_cycle_analysis(simple)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached cycle analysis to {cache_path}")
    return result


def plot_suspicious_cluster(G: nx.MultiDiGraph, illicit_cycles: set, communities: list,
                             illicit_nodes: set):
    if illicit_cycles:
        # Pick longest, then alphabetical, so the result is stable.
        cluster = max(illicit_cycles, key=lambda c: (len(c), c))
        sub_nodes = list(cluster)
        title = f"Detected {len(cluster)}-account cycle containing a laundering transaction"
    else:
        comm_illicit_counts = [
            (i, len(comm & illicit_nodes)) for i, comm in enumerate(communities)
        ]
        best_comm_idx, _ = max(comm_illicit_counts, key=lambda x: x[1])
        comm_illicit_nodes = communities[best_comm_idx] & illicit_nodes
        sub_nodes = set(comm_illicit_nodes)
        for n in list(comm_illicit_nodes):
            sub_nodes.update(G.predecessors(n))
            sub_nodes.update(G.successors(n))
        sub_nodes = list(sub_nodes)[:40]
        title = f"Densest illicit community (community #{best_comm_idx}, sampled to {len(sub_nodes)} nodes)"

    sub = G.subgraph(sub_nodes)
    sub_simple = nx.DiGraph()
    sub_simple.add_nodes_from(sub.nodes())
    for u, v, is_laundering in sub.edges(data="is_laundering"):
        if sub_simple.has_edge(u, v):
            if is_laundering == 1:
                sub_simple[u][v]["is_laundering"] = 1
        else:
            sub_simple.add_edge(u, v, is_laundering=is_laundering)

    if illicit_cycles:
        # Lay out nodes in cycle order
        angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(sub_nodes), endpoint=False)
        pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(sub_nodes, angles)}
    else:
        pos = nx.spring_layout(sub_simple, seed=42)
    node_colors = ["#c0392b" if n in illicit_nodes else "steelblue" for n in sub_simple.nodes()]
    loop_edges = [(u, v) for u, v in sub_simple.edges() if u == v]
    if illicit_cycles:
        cycle_edges = {(sub_nodes[i], sub_nodes[(i + 1) % len(sub_nodes)]) for i in range(len(sub_nodes))}
    else:
        cycle_edges = {(u, v) for u, v in sub_simple.edges() if u != v}
    flow_edges = [e for e in sub_simple.edges() if e in cycle_edges]
    other_edges = [(u, v) for u, v in sub_simple.edges() if u != v and (u, v) not in cycle_edges]
    flow_colors = [
        "#c0392b" if sub_simple[u][v].get("is_laundering") == 1 else "dimgray" for u, v in flow_edges
    ]

    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_nodes(sub_simple, pos, ax=ax, node_color=node_colors, node_size=700, alpha=0.9)
    nx.draw_networkx_edges(sub_simple, pos, ax=ax, edgelist=other_edges, edge_color="silver",
                           style="dotted", arrows=True, arrowsize=14, width=1.2, node_size=700)
    nx.draw_networkx_edges(sub_simple, pos, ax=ax, edgelist=flow_edges, edge_color=flow_colors,
                           arrows=True, arrowsize=24, width=3.0, node_size=700)
    # Put labels just outside each node
    for n, (x, y) in pos.items():
        r = (x * x + y * y) ** 0.5 or 1.0
        lx, ly = x + 0.22 * x / r, y + 0.22 * y / r
        ha = "left" if lx > 0.05 else "right" if lx < -0.05 else "center"
        ax.text(lx, ly, n, fontsize=12, ha=ha, va="center", family="monospace")
    n_loops = len(loop_edges)
    handles = [mpatches.Patch(color="#c0392b", label="Illicit-labelled account / laundering transaction")]
    if any(n not in illicit_nodes for n in sub_simple.nodes()):
        handles.append(mpatches.Patch(color="steelblue", label="Licit-labelled account"))
    if "dimgray" in flow_colors:
        handles.append(mpatches.Patch(color="dimgray", label="Non-laundering transaction in the cycle"))
    if other_edges:
        handles.append(Line2D([0], [0], color="silver", linestyle=":", linewidth=2,
                              label="Other transfers among these accounts"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=1,
              fontsize=11, frameon=False)
    if n_loops:
        ax.text(0, -1.38, f"Self-transfers on {n_loops} of {len(sub_nodes)} accounts omitted for clarity",
                ha="center", fontsize=10, style="italic")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.5, 1.35)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(CLUSTER_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    return title


def main() -> None:
    t_start = time.time()

    print("Loading graph...")
    G = load_graph()
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} multi-edges")

    print("Building simple DiGraph (collapsed multi-edges)...")
    simple = build_simple_digraph(G)
    print(f"  Simple DiGraph: {simple.number_of_edges():,} unique edges")

    print("Identifying illicit-labeled nodes/edges...")
    illicit_nodes, illicit_edge_pairs = get_illicit_nodes_and_edges(G)
    print(f"  {len(illicit_nodes):,} illicit-touching accounts, {len(illicit_edge_pairs):,} illicit (u,v) pairs")

    # Betweenness (approximate) and Louvain communities (cached)
    betweenness, communities = get_graph_metrics_cached(simple)

    top20_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
    top20_illicit_count = sum(1 for acct, _ in top20_betweenness if acct in illicit_nodes)

    node_to_comm = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = i

    comm_sizes = sorted((len(c) for c in communities), reverse=True)
    illicit_comm_ids = [node_to_comm[n] for n in illicit_nodes if n in node_to_comm]
    illicit_comm_counter = {}
    for cid in illicit_comm_ids:
        illicit_comm_counter[cid] = illicit_comm_counter.get(cid, 0) + 1
    n_distinct_illicit_comms = len(illicit_comm_counter)
    largest_illicit_comm_id, largest_illicit_comm_count = max(
        illicit_comm_counter.items(), key=lambda x: x[1]
    )
    pct_illicit_in_largest = 100 * largest_illicit_comm_count / len(illicit_comm_ids)

    # Cycle detection (cached)
    print("Running cycle detection (cached if available)...")
    t0 = time.time()
    cycle_result = get_cycle_analysis_cached(simple)
    print(f"  Done in {time.time() - t0:.1f}s")

    two_cycles = cycle_result["two_cycles"]
    three_cycles = cycle_result["three_cycles"]
    small_scc_cycles = cycle_result["small_scc_cycles"]
    giant_scc_cycles = cycle_result["giant_scc_cycles"]
    all_cycles = cycle_result["all_cycles"]
    nontrivial_sccs_count = cycle_result["nontrivial_scc_count"]
    giant_scc_size = cycle_result["giant_scc_size"]
    small_sccs_count = cycle_result["small_scc_count"]
    small_sccs_max_size = cycle_result["small_scc_max_size"]

    print(f"  2-cycles: {len(two_cycles)}, 3-cycles: {len(three_cycles)}")
    print(f"  {nontrivial_sccs_count} non-trivial SCCs; giant SCC size={giant_scc_size:,}, "
          f"{small_sccs_count} small SCCs (max size {small_sccs_max_size})")
    print(f"  4-6 length cycles in small SCCs: {len(small_scc_cycles)}")
    print(f"  4-6 length cycles in giant SCC (exact): {len(giant_scc_cycles)}")

    illicit_cycles = {c for c in all_cycles if cycle_touches_illicit(c, illicit_edge_pairs)}
    pct_illicit_cycles = 100 * len(illicit_cycles) / len(all_cycles) if all_cycles else 0.0

    # Visualization
    print("Rendering suspicious cluster visualization...")
    viz_title = plot_suspicious_cluster(G, illicit_cycles, communities, illicit_nodes)
    print(f"  Saved {CLUSTER_PNG} ({viz_title})")

    # Write summary
    lines = [
        "# Graph Analysis Summary",
        "",
        "## Scope and methodology notes",
        f"- Betweenness centrality: APPROXIMATE, k={BETWEENNESS_K} random-source sampling "
        "(exact is infeasible at 515K nodes).",
        "- Cycle detection (length 2-6): EXACT for 2- and 3-cycles (whole graph) and for "
        f"4-6 length cycles within the {small_sccs_count} small strongly-connected components "
        f"(max size {small_sccs_max_size}).",
        f"- Cycle detection within the giant {giant_scc_size:,}-node strongly connected "
        "component: exact (exhaustive depth-limited search).",
        "",
        "## Betweenness centrality — top 20 accounts",
        f"- Of the top 20 accounts by (approximate) betweenness centrality, "
        f"{top20_illicit_count} are labeled illicit (touch at least one illicit transaction).",
        "",
        "| Rank | Account | Betweenness (approx) | Illicit? |",
        "|---|---|---|---|",
    ]
    for rank, (acct, score) in enumerate(top20_betweenness, start=1):
        flag = "yes" if acct in illicit_nodes else ""
        lines.append(f"| {rank} | {acct} | {score:.6f} | {flag} |")

    lines += [
        "",
        "## Community detection (Louvain, undirected; weight 2 if money moved both ways, else 1)",
        f"- Number of communities found: {len(communities):,}",
        f"- Community size distribution — largest: {comm_sizes[0]:,}, "
        f"median: {comm_sizes[len(comm_sizes)//2]:,}, smallest: {comm_sizes[-1]:,}",
        f"- Top 10 community sizes: {comm_sizes[:10]}",
        f"- Illicit accounts fall into {n_distinct_illicit_comms:,} distinct communities "
        f"(out of {len(illicit_comm_ids):,} illicit accounts placed in any community).",
        f"- {pct_illicit_in_largest:.2f}% of illicit accounts fall into the single largest "
        f"illicit-containing community (community #{largest_illicit_comm_id}, "
        f"{largest_illicit_comm_count:,} illicit accounts) — the remainder are spread "
        f"across the other {n_distinct_illicit_comms - 1:,} communities.",
        "",
        "## Cycle detection (directed cycles, length 2-6)",
        f"- 2-cycles (exact, whole graph): {len(two_cycles):,}",
        f"- 3-cycles (exact, whole graph): {len(three_cycles):,}",
        f"- 4-6 length cycles in small SCCs (exact): {len(small_scc_cycles):,}",
        f"- 4-6 length cycles in giant SCC (exact): {len(giant_scc_cycles):,}",
        f"- **Total cycles found: {len(all_cycles):,}**",
        f"- Cycles involving at least one illicit-labeled edge: {len(illicit_cycles):,} "
        f"({pct_illicit_cycles:.2f}% of all cycles found)",
        "",
        "## Visualization",
        f"- Saved to `notebooks/suspicious_cluster.png`: {viz_title}",
        "",
        f"Total analysis runtime: {time.time() - t_start:.1f}s",
        "",
    ]
    summary = "\n".join(lines)
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
