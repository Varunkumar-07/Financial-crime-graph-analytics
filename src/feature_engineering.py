"""
Builds the per-account feature table: graph features (degree,
betweenness, community, cycles) plus transaction totals.

- No velocity feature: it's just transaction_count rescaled.
- Graph metrics reuse graph_analysis.py (same settings).
- community_illicit_density uses training labels only (out-of-fold for
  training accounts), so no account sees its own label.
- Cycle counts come from the shared cycle cache.

Usage: python src/feature_engineering.py
"""

import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import graph_analysis as ga  # reuse graph construction / centrality / cycle logic
from common import FEATURES_CSV, PROJECT_ROOT, load_transactions
from splits import RANDOM_STATE, split_indices

REPORT_MD = PROJECT_ROOT / "notebooks" / "feature_engineering_summary.md"


def compute_transaction_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    sent = (
        df.groupby("From Account")["Amount Paid"]
        .agg(total_sent="sum", sent_count="count")
        .reset_index()
        .rename(columns={"From Account": "account"})
    )
    received = (
        df.groupby("To Account")["Amount Received"]
        .agg(total_received="sum", received_count="count")
        .reset_index()
        .rename(columns={"To Account": "account"})
    )
    agg = sent.merge(received, on="account", how="outer")
    agg[["total_sent", "sent_count", "total_received", "received_count"]] = agg[
        ["total_sent", "sent_count", "total_received", "received_count"]
    ].fillna(0)

    agg["transaction_count"] = agg["sent_count"] + agg["received_count"]
    agg["avg_transaction_amount"] = (
        agg["total_sent"] + agg["total_received"]
    ) / agg["transaction_count"]
    agg["in_out_amount_ratio"] = np.where(
        agg["total_sent"] > 0, agg["total_received"] / agg["total_sent"], np.nan
    )
    return agg[
        [
            "account",
            "total_sent",
            "total_received",
            "transaction_count",
            "avg_transaction_amount",
            "in_out_amount_ratio",
        ]
    ]


def compute_cycle_participation(simple: nx.DiGraph):
    analysis = ga.get_cycle_analysis_cached(simple)
    all_cycles = analysis["all_cycles"]
    cycle_count = {}
    for cyc in all_cycles:
        for node in cyc:
            cycle_count[node] = cycle_count.get(node, 0) + 1
    return cycle_count, len(all_cycles)


def train_only_community_density(community_id: np.ndarray, is_illicit: np.ndarray,
                                  train_idx: np.ndarray, n_folds: int = 5) -> np.ndarray:
    """Illicit share of training accounts in the same community.

    Val/test accounts use all training labels; training accounts use
    out-of-fold values (leave-one-out would leak the label).
    Empty communities fall back to the overall training rate.
    """
    n_comm = int(community_id.max()) + 1
    prior = is_illicit[train_idx].mean()

    def density_from(source_idx, target_idx):
        cnt = np.bincount(community_id[source_idx], minlength=n_comm).astype(float)
        pos = np.bincount(community_id[source_idx], weights=is_illicit[source_idx].astype(float),
                          minlength=n_comm)
        c = community_id[target_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cnt[c] > 0, pos[c] / cnt[c], prior)

    density = density_from(train_idx, np.arange(len(community_id)))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    for fit_pos, hold_pos in folds.split(train_idx, is_illicit[train_idx]):
        density[train_idx[hold_pos]] = density_from(train_idx[fit_pos], train_idx[hold_pos])
    return density


def main() -> None:
    t_start = time.time()

    print("Loading raw transaction data...")
    df = load_transactions(parse_timestamps=True)

    print("Loading graph...")
    G = ga.load_graph()
    simple = ga.build_simple_digraph(G)
    illicit_nodes, _ = ga.get_illicit_nodes_and_edges(G)
    print(f"  {simple.number_of_nodes():,} nodes, {simple.number_of_edges():,} simple edges, "
          f"{len(illicit_nodes):,} illicit-touching accounts")

    print(f"Computing betweenness centrality (k={ga.BETWEENNESS_K}, approximate)...")
    t0 = time.time()
    betweenness = nx.betweenness_centrality(simple, k=ga.BETWEENNESS_K, seed=42, normalized=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Computing Louvain communities...")
    t0 = time.time()
    undirected = ga.build_undirected_weighted(simple)
    communities = nx.community.louvain_communities(undirected, weight="weight", seed=42)
    print(f"  Done in {time.time() - t0:.1f}s, {len(communities)} communities")

    node_to_comm = {}
    comm_size = {}
    for i, comm in enumerate(communities):
        comm_size[i] = len(comm)
        for n in comm:
            node_to_comm[n] = i

    print("Computing cycle participation...")
    cycle_count, n_total_cycles = compute_cycle_participation(simple)

    print("Computing transaction-level aggregates...")
    agg = compute_transaction_aggregates(df)

    total_days = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 86400
    print(f"  Observed window: {total_days:.3f} days")

    print("Assembling feature table...")
    accounts = list(simple.nodes())
    features = pd.DataFrame({"account": accounts})

    in_deg = dict(simple.in_degree())
    out_deg = dict(simple.out_degree())
    features["in_degree"] = features["account"].map(in_deg)
    features["out_degree"] = features["account"].map(out_deg)
    features["total_degree"] = features["in_degree"] + features["out_degree"]

    features["betweenness"] = features["account"].map(betweenness)

    features["community_id"] = features["account"].map(node_to_comm)
    features["community_size"] = features["community_id"].map(comm_size)
    # Filled in below once labels and split are ready.
    features["community_illicit_density"] = np.nan

    features["cycle_count"] = features["account"].map(cycle_count).fillna(0).astype(int)
    features["cycle_flag"] = (features["cycle_count"] > 0).astype(int)

    features = features.merge(agg, on="account", how="left")
    features[["total_sent", "total_received", "transaction_count"]] = features[
        ["total_sent", "total_received", "transaction_count"]
    ].fillna(0)

    features["is_illicit"] = features["account"].isin(illicit_nodes).astype(int)

    train_idx, _, _ = split_indices(features["is_illicit"].values)
    features["community_illicit_density"] = train_only_community_density(
        features["community_id"].values, features["is_illicit"].values, train_idx
    )

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)
    print(f"Saved feature table to {FEATURES_CSV}")

    # --- Sanity checks ---
    n_rows, n_cols = features.shape
    null_counts = features.isnull().sum()
    null_counts = null_counts[null_counts > 0]

    negative_checks = {
        "total_sent < 0": (features["total_sent"] < 0).sum(),
        "total_received < 0": (features["total_received"] < 0).sum(),
        "in_degree < 0": (features["in_degree"] < 0).sum(),
        "out_degree < 0": (features["out_degree"] < 0).sum(),
    }
    inf_counts = np.isinf(features.select_dtypes(include=[np.number])).sum()
    inf_counts = inf_counts[inf_counts > 0]

    class_counts = features["is_illicit"].value_counts().sort_index()
    class_pct = features["is_illicit"].value_counts(normalize=True).sort_index() * 100

    # --- Correlation with label ---
    numeric_cols = [
        "in_degree", "out_degree", "total_degree", "betweenness",
        "community_size", "community_illicit_density", "cycle_count", "cycle_flag",
        "total_sent", "total_received", "transaction_count",
        "avg_transaction_amount", "in_out_amount_ratio",
    ]
    corr = features[numeric_cols + ["is_illicit"]].corr()["is_illicit"].drop("is_illicit")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

    group_means = features.groupby("is_illicit")[numeric_cols].mean().T
    group_means.columns = ["licit_mean", "illicit_mean"]

    # --- Report ---
    lines = [
        "# Feature Engineering Summary",
        "",
        "## Methodology notes",
        f"- Betweenness centrality: APPROXIMATE, k={ga.BETWEENNESS_K} random-source sampling "
        "(recomputed with the same parameters as graph_analysis.py).",
        "- Louvain communities: recomputed with seed=42 on the undirected, "
        "transaction-count-weighted graph.",
        "- community_illicit_density: computed from TRAINING-split labels only (5-fold "
        "out-of-fold for training accounts, so no account's own label enters its own value); "
        "communities with no labelled training members use the global training illicit rate. "
        "No validation/test label enters any feature.",
        f"- Cycle participation (length 2-6): {n_total_cycles:,} total cycles (shared cache) — exact "
        "for 2/3-cycles and small-SCC 4-6 cycles, PARTIAL SAMPLE (120s cap) for the giant "
        "17,075-node SCC. cycle_count is therefore a lower bound for accounts in that component.",
        f"- Observed dataset window: {total_days:.3f} days (computed directly from min/max "
        "Timestamp). A `velocity` feature (transaction_count / window) was considered but "
        "dropped: since the window is constant across all accounts, it's a pure linear "
        "rescaling of `transaction_count` — perfectly collinear, adds no information, and "
        "was found to distort tree-based feature importance rankings (see baseline_model_summary.md).",
        "",
        "## Feature table shape",
        f"- {n_rows:,} rows x {n_cols} columns",
        "",
        "## Class balance (is_illicit)",
    ]
    for label in class_counts.index:
        name = "Illicit (1)" if label == 1 else "Licit (0)"
        lines.append(f"- {name}: {class_counts[label]:,} ({class_pct[label]:.4f}%)")

    lines += ["", "## Sanity checks", "### Missing values (NaN)"]
    if len(null_counts) == 0:
        lines.append("- None")
    else:
        for col, cnt in null_counts.items():
            note = (
                " (expected: accounts with zero total_sent have an undefined "
                "in_out_amount_ratio)" if col == "in_out_amount_ratio" else ""
            )
            lines.append(f"- `{col}`: {cnt:,}{note}")

    lines.append("### Negative-value checks")
    for check, cnt in negative_checks.items():
        lines.append(f"- {check}: {cnt:,}")

    lines.append("### Infinite-value checks")
    if len(inf_counts) == 0:
        lines.append("- None")
    else:
        for col, cnt in inf_counts.items():
            lines.append(f"- `{col}`: {cnt:,}")

    lines += [
        "",
        "## Correlation with is_illicit (sorted by |correlation|)",
        "",
        "| Feature | Correlation | Licit mean | Illicit mean |",
        "|---|---|---|---|",
    ]
    for feat in corr.index:
        # .6g so tiny values (e.g. betweenness) don't show as 0.0000.
        lines.append(
            f"| {feat} | {corr[feat]:.6g} | {group_means.loc[feat, 'licit_mean']:.6g} | "
            f"{group_means.loc[feat, 'illicit_mean']:.6g} |"
        )

    lines += ["", f"Total runtime: {time.time() - t_start:.1f}s", ""]
    summary = "\n".join(lines)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
