"""
Builds the transaction graph from HI-Small_Trans.csv.
Accounts are nodes, transactions are directed edges (MultiDiGraph,
since accounts can transact more than once).

Usage: python src/build_graph.py
"""

import pickle
import time

import networkx as nx
import pandas as pd

from common import PROJECT_ROOT, load_transactions

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "transaction_graph.gpickle"
STATS_MD = PROJECT_ROOT / "notebooks" / "graph_stats.md"

LAUNDERING_COL = "Is Laundering"


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    # Use the sender-side amount/currency as the edge values.
    edge_df = df.rename(
        columns={
            "Timestamp": "timestamp",
            "Amount Paid": "amount",
            "Payment Currency": "currency",
            LAUNDERING_COL: "is_laundering",
        }
    )
    graph = nx.from_pandas_edgelist(
        edge_df,
        source="From Account",
        target="To Account",
        edge_attr=["timestamp", "amount", "currency", "is_laundering"],
        create_using=nx.MultiDiGraph(),
    )
    return graph


def compute_stats(graph: nx.MultiDiGraph, df: pd.DataFrame) -> str:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    avg_in_degree = n_edges / n_nodes
    avg_out_degree = n_edges / n_nodes

    wcc = list(nx.weakly_connected_components(graph))
    n_wcc = len(wcc)
    largest_wcc = max(wcc, key=len)

    illicit_edges = df[df[LAUNDERING_COL] == 1]
    n_illicit_edges = len(illicit_edges)
    illicit_nodes = set(illicit_edges["From Account"]) | set(illicit_edges["To Account"])
    n_illicit_nodes = len(illicit_nodes)

    lines = [
        "# Graph Stats — Transaction MultiDiGraph",
        "",
        "## Size",
        f"- Total nodes (accounts): {n_nodes:,}",
        f"- Total edges (transactions): {n_edges:,}",
        "",
        "## Degree",
        f"- Average in-degree: {avg_in_degree:.4f}",
        f"- Average out-degree: {avg_out_degree:.4f}",
        "",
        "## Connectivity (weakly connected components)",
        f"- Number of weakly connected components: {n_wcc:,}",
        f"- Largest weakly connected component size (nodes): {len(largest_wcc):,}",
        f"- Largest weakly connected component as % of all nodes: {100 * len(largest_wcc) / n_nodes:.2f}%",
        "",
        "## Illicit involvement",
        f"- Edges (transactions) labeled illicit: {n_illicit_edges:,}",
        f"- Nodes (accounts) involved in at least one illicit transaction: {n_illicit_nodes:,} "
        f"({100 * n_illicit_nodes / n_nodes:.4f}% of all nodes)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    print("Loading data...")
    df = load_transactions()
    print(f"  {len(df):,} rows loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("Building graph...")
    graph = build_graph(df)
    print(f"  Graph built in {time.time() - t1:.1f}s")

    t2 = time.time()
    print("Computing stats...")
    stats = compute_stats(graph, df)
    print(f"  Stats computed in {time.time() - t2:.1f}s")
    print(stats)

    STATS_MD.parent.mkdir(parents=True, exist_ok=True)
    STATS_MD.write_text(stats)

    t3 = time.time()
    print("Saving graph...")
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Graph saved to {GRAPH_PATH} in {time.time() - t3:.1f}s")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
