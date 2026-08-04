"""
Graph analysis on the transaction MultiDiGraph built by build_graph.py:
degree/betweenness centrality, Louvain community detection, and bounded
directed-cycle detection.

Scale notes (515K nodes / 5.08M multi-edges, collapsed to ~1.02M unique
directed edges):
  - Betweenness centrality is APPROXIMATED via k=500 random source
    samples (exact Brandes betweenness is infeasible at this scale;
    timed at ~4.7s + 0.39s/sample during development, so k=500 ~ 3-4 min).
  - Cycle detection (length 2-6) is EXACT everywhere except one giant
    17,075-node strongly connected component:
      * 2-cycles and 3-cycles: found exactly, across the whole graph,
        via direct edge/triangle lookups (polynomial, not exponential).
      * 4-6 length cycles in the 2,927 small SCCs (max size 12): exact,
        via nx.simple_cycles(length_bound=6).
      * 4-6 length cycles in the giant SCC: nx.simple_cycles is a
        generator; a full run hung for 5.5+ minutes with no result even
        at length_bound=2 (confirmed and killed during development), so
        it is instead consumed under a hard 120-second wall-clock cap
        and the result is reported as a PARTIAL SAMPLE (lower bound),
        not an exhaustive count.

No feature engineering or modeling here — analysis and reporting only.

Run from the project root with the venv activated:

    python src/graph_analysis.py
"""

import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "transaction_graph.gpickle"
SUMMARY_MD = PROJECT_ROOT / "notebooks" / "graph_analysis_summary.md"
CLUSTER_PNG = PROJECT_ROOT / "notebooks" / "suspicious_cluster.png"

BETWEENNESS_K = 500
GIANT_SCC_TIME_BUDGET_SEC = 120


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
        for w in simple.successors(v):
            if w != u and w != v and (w, u) in edge_set:
                three.add(canonicalize_cycle([u, v, w]))
    return two, three


def find_small_scc_cycles(simple: nx.DiGraph, small_sccs: list):
    found = set()
    for scc in small_sccs:
        sub = simple.subgraph(scc)
        for cyc in nx.simple_cycles(sub, length_bound=6):
            if len(cyc) >= 4:
                found.add(canonicalize_cycle(cyc))
    return found


def find_giant_scc_cycles_sampled(simple: nx.DiGraph, giant_scc: set, time_budget: float):
    sub = simple.subgraph(giant_scc).copy()
    found = set()
    deadline = time.time() + time_budget
    hit_cap = False
    for cyc in nx.simple_cycles(sub, length_bound=6):
        if len(cyc) >= 4:
            found.add(canonicalize_cycle(cyc))
        if time.time() > deadline:
            hit_cap = True
            break
    return found, hit_cap


CYCLES_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "all_cycles_cache.pkl"


def compute_cycle_analysis(simple: nx.DiGraph) -> dict:
    """Runs the full cycle-detection pipeline (2/3-cycles exact, small-SCC
    4-6 cycles exact, giant-SCC 4-6 cycles time-capped partial sample —
    see module docstring) and returns every piece needed both for the
    detailed report here and for simpler downstream consumers (the flat
    `all_cycles` set used by explainability.py / the API)."""
    two_cycles, three_cycles = find_2_and_3_cycles(simple)

    sccs = list(nx.strongly_connected_components(simple))
    nontrivial_sccs = [s for s in sccs if len(s) > 1]
    giant_scc = max(nontrivial_sccs, key=len)
    small_sccs = [s for s in nontrivial_sccs if s is not giant_scc]

    small_scc_cycles = find_small_scc_cycles(simple, small_sccs)
    giant_scc_cycles, hit_cap = find_giant_scc_cycles_sampled(simple, giant_scc, GIANT_SCC_TIME_BUDGET_SEC)

    all_exact_cycles = two_cycles | three_cycles | small_scc_cycles
    all_cycles = all_exact_cycles | giant_scc_cycles

    return {
        "two_cycles": two_cycles,
        "three_cycles": three_cycles,
        "small_scc_cycles": small_scc_cycles,
        "giant_scc_cycles": giant_scc_cycles,
        "hit_cap": hit_cap,
        "all_exact_cycles": all_exact_cycles,
        "all_cycles": all_cycles,
        "nontrivial_scc_count": len(nontrivial_sccs),
        "giant_scc_size": len(giant_scc),
        "small_scc_count": len(small_sccs),
        "small_scc_max_size": max(len(s) for s in small_sccs),
    }


def get_cycle_analysis_cached(simple: nx.DiGraph, cache_path=CYCLES_CACHE_PATH) -> dict:
    """Disk-cached wrapper around compute_cycle_analysis(). Recomputing the
    full detection (~150-300s, dominated by the giant-SCC time-capped
    search) on every run/every process that needs it (graph_analysis.py,
    explainability.py, the API's own startup) is wasteful since the result
    is deterministic for a fixed graph — computed once, reused after that.

    Detects and transparently discards the older cache format (a bare
    `set`, from before this function existed) rather than erroring, so a
    stale cache file never needs manual deletion.
    """
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and "all_cycles" in cached:
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
                             node_to_comm: dict, illicit_nodes: set):
    if illicit_cycles:
        cluster = max(illicit_cycles, key=len)
        sub_nodes = list(cluster)
        title = f"Suspicious cycle ({len(cluster)}-cycle, touches an illicit transaction)"
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

    pos = nx.spring_layout(sub_simple, seed=42)
    node_colors = ["red" if n in illicit_nodes else "steelblue" for n in sub_simple.nodes()]
    edge_colors = [
        "red" if d.get("is_laundering") == 1 else "gray"
        for _, _, d in sub_simple.edges(data=True)
    ]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(sub_simple, pos, node_color=node_colors, node_size=400, alpha=0.85)
    nx.draw_networkx_edges(
        sub_simple, pos, edge_color=edge_colors, arrows=True, arrowsize=15, width=1.5
    )
    nx.draw_networkx_labels(sub_simple, pos, font_size=7)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(CLUSTER_PNG, dpi=150)
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

    # --- Degree centrality (cheap, exact) ---
    print("Computing degree centrality...")
    in_deg_cent = nx.in_degree_centrality(G)
    out_deg_cent = nx.out_degree_centrality(G)

    # --- Betweenness centrality (approximate) ---
    print(f"Computing approximate betweenness centrality (k={BETWEENNESS_K})...")
    t0 = time.time()
    betweenness = nx.betweenness_centrality(simple, k=BETWEENNESS_K, seed=42, normalized=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    top20_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
    top20_illicit_count = sum(1 for acct, _ in top20_betweenness if acct in illicit_nodes)

    # --- Louvain community detection ---
    print("Building undirected weighted graph for Louvain...")
    undirected = build_undirected_weighted(simple)
    print("Running Louvain community detection...")
    t0 = time.time()
    communities = nx.community.louvain_communities(undirected, weight="weight", seed=42)
    print(f"  Done in {time.time() - t0:.1f}s, {len(communities)} communities")

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

    # --- Cycle detection (disk-cached — see get_cycle_analysis_cached docstring) ---
    print("Running cycle detection (cached if available)...")
    t0 = time.time()
    cycle_result = get_cycle_analysis_cached(simple)
    print(f"  Done in {time.time() - t0:.1f}s")

    two_cycles = cycle_result["two_cycles"]
    three_cycles = cycle_result["three_cycles"]
    small_scc_cycles = cycle_result["small_scc_cycles"]
    giant_scc_cycles = cycle_result["giant_scc_cycles"]
    hit_cap = cycle_result["hit_cap"]
    all_exact_cycles = cycle_result["all_exact_cycles"]
    all_cycles = cycle_result["all_cycles"]
    nontrivial_sccs_count = cycle_result["nontrivial_scc_count"]
    giant_scc_size = cycle_result["giant_scc_size"]
    small_sccs_count = cycle_result["small_scc_count"]
    small_sccs_max_size = cycle_result["small_scc_max_size"]

    print(f"  2-cycles: {len(two_cycles)}, 3-cycles: {len(three_cycles)}")
    print(f"  {nontrivial_sccs_count} non-trivial SCCs; giant SCC size={giant_scc_size:,}, "
          f"{small_sccs_count} small SCCs (max size {small_sccs_max_size})")
    print(f"  4-6 length cycles in small SCCs: {len(small_scc_cycles)}")
    print(f"  4-6 length cycles in giant SCC: {len(giant_scc_cycles)} "
          f"({'CAPPED — partial sample' if hit_cap else 'exhausted before cap'})")

    illicit_cycles = {c for c in all_cycles if cycle_touches_illicit(c, illicit_edge_pairs)}
    pct_illicit_cycles = 100 * len(illicit_cycles) / len(all_cycles) if all_cycles else 0.0

    # --- Visualization ---
    print("Rendering suspicious cluster visualization...")
    viz_title = plot_suspicious_cluster(G, illicit_cycles, communities, node_to_comm, illicit_nodes)
    print(f"  Saved {CLUSTER_PNG} ({viz_title})")

    # --- Write summary ---
    giant_scc_note = (
        "cap was hit — this is a lower bound, not exhaustive"
        if hit_cap
        else "search completed before the cap"
    )
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
        f"component is a PARTIAL SAMPLE: a full run hung for 5.5+ minutes with no result "
        f"and was killed during development, so it was capped at "
        f"{GIANT_SCC_TIME_BUDGET_SEC}s wall-clock ({giant_scc_note}).",
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
        "## Community detection (Louvain, undirected, weighted by transaction count)",
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
        f"- 4-6 length cycles in giant SCC ({GIANT_SCC_TIME_BUDGET_SEC}s sample, "
        f"{'PARTIAL — lower bound' if hit_cap else 'exhaustive'}): {len(giant_scc_cycles):,}",
        f"- **Total cycles found: {len(all_cycles):,}** "
        f"({len(all_exact_cycles):,} exact + {len(giant_scc_cycles):,} sampled)",
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
