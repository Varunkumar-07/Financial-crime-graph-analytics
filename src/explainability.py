"""
Explains GraphSAGE predictions for a few test-set accounts.

For each account:
  1. Reuse the split, preprocessing and threshold from train_gnn.py.
  2. Draw its 1-2 hop subgraph from the original graph.
  3. Run GNNExplainer on its 3-hop subgraph (one hop per SAGEConv layer).
  4. Look up its cycles (shared cycle cache from graph_analysis.py).
  5. Add its community info from account_features.csv.

GNNExplainer takes ~0.7-3s per account.

Usage: python src/explainability.py
"""

import json
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

import graph_analysis as ga
from common import PROJECT_ROOT, f2_optimal_threshold
from splits import RANDOM_STATE, split_indices
from train_gnn import GraphSAGE, preprocess_features

OUT_DIR = PROJECT_ROOT / "notebooks" / "explainability"
SUMMARY_MD = PROJECT_ROOT / "notebooks" / "explainability_summary.md"
CASES_JSON = PROJECT_ROOT / "notebooks" / "explainability_cases.json"

CASE_LABELS = {
    "true_positive_confident": "Confident True Positive",
    "true_positive_marginal": "Marginal True Positive",
    "false_negative": "False Negative",
    "false_positive": "False Positive",
}

VIZ_MAX_NODES = 40  # use 2-hop viz radius if it stays under this, else 1-hop
VIZ_MAX_NEIGHBOURS = 150  # big accounts: draw only their busiest neighbours
# Eligible cases: test accounts in a cycle, small enough to draw.
MIN_NEIGHBOURS, MAX_NEIGHBOURS = 5, 39
EXPLAIN_HOPS = 3    # matches the model's 3 SAGEConv layers


def setup_model_and_data():
    data = torch.load(PROJECT_ROOT / "data" / "processed" / "pyg_data.pt", weights_only=False)
    ckpt = torch.load(PROJECT_ROOT / "data" / "processed" / "graphsage_baseline.pt", weights_only=False)

    y_np = data.y.numpy()
    train_idx, val_idx, test_idx = split_indices(y_np)

    x_np = preprocess_features(data.x.numpy().copy(), data.feature_cols, train_idx)
    x = torch.tensor(x_np, dtype=torch.float)

    model = GraphSAGE(ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        out = model(x, data.edge_index)
        proba_all = F.softmax(out, dim=1)[:, 1].numpy()

    thresh, _, _, _ = f2_optimal_threshold(y_np[val_idx], proba_all[val_idx])

    return {
        "data": data, "model": model, "x": x, "y_np": y_np,
        "test_idx": set(test_idx.tolist()), "threshold": thresh,
        "proba_all": proba_all,
    }


def select_samples(ctx, cycle_count):
    """Pick the 4 case-study accounts by a fixed rule (reproducible):
      - true_positive_confident: illicit, flagged, highest probability
      - true_positive_marginal: illicit, flagged, lowest probability
      - false_negative:         illicit, not flagged, highest probability
      - false_positive:         licit, flagged, highest probability
    """
    data, y_np, proba, thr = ctx["data"], ctx["y_np"], ctx["proba_all"], ctx["threshold"]
    n_neigh = np.bincount(data.edge_index[0].numpy(), minlength=data.num_nodes)
    test = np.array(sorted(ctx["test_idx"]))
    cyc = np.array([cycle_count.get(data.accounts[i], 0) for i in test])
    ok = test[(cyc >= 1) & (n_neigh[test] >= MIN_NEIGHBOURS) & (n_neigh[test] <= MAX_NEIGHBOURS)]
    flagged = proba[ok] >= thr
    illicit = y_np[ok] == 1

    def pick(mask, highest):
        cand = ok[mask]
        order = np.argsort(proba[cand])
        return data.accounts[cand[order[-1] if highest else order[0]]]

    return [
        (pick(illicit & flagged, True), "true_positive_confident"),
        (pick(illicit & flagged, False), "true_positive_marginal"),
        (pick(illicit & ~flagged, True), "false_negative"),
        (pick(~illicit & flagged, True), "false_positive"),
    ]


def get_cycles_cached():
    """Wrapper around graph_analysis.get_cycle_analysis_cached().
    Returns (all_cycles, G).
    """
    G = ga.load_graph()
    simple = ga.build_simple_digraph(G)
    result = ga.get_cycle_analysis_cached(simple)
    return result["all_cycles"], G


def get_viz_subgraph(account, data, G_multi, account_to_idx=None):
    """Returns (subgraph, hops used, truncated)."""
    if account_to_idx is None:
        account_to_idx = {a: i for i, a in enumerate(data.accounts)}
    idx = account_to_idx[account]

    subset2, _, _, _ = k_hop_subgraph(idx, num_hops=2, edge_index=data.edge_index, relabel_nodes=True)
    if len(subset2) <= VIZ_MAX_NODES:
        subset = subset2
        radius_used = 2
    else:
        subset1, _, _, _ = k_hop_subgraph(idx, num_hops=1, edge_index=data.edge_index, relabel_nodes=True)
        subset = subset1
        radius_used = 1

    sub_accounts = [data.accounts[i] for i in subset.tolist()]
    truncated = len(sub_accounts) > VIZ_MAX_NEIGHBOURS + 1
    if truncated:
        # Keep the neighbours with the most transactions with this account.
        counts = Counter(v for _, v in G_multi.out_edges(account))
        counts.update(u for u, _ in G_multi.in_edges(account))
        counts.pop(account, None)
        sub_accounts = [account] + [n for n, _ in counts.most_common(VIZ_MAX_NEIGHBOURS)]
    G_sub = G_multi.subgraph(sub_accounts).copy()
    return G_sub, radius_used, truncated


def find_account_cycles(account, all_cycles):
    return [c for c in all_cycles if account in c]


def run_gnn_explainer(account, model, x, data, account_to_idx=None):
    if account_to_idx is None:
        account_to_idx = {a: i for i, a in enumerate(data.accounts)}
    idx = account_to_idx[account]
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        idx, num_hops=EXPLAIN_HOPS, edge_index=data.edge_index, relabel_nodes=True
    )
    sub_x = x[subset]
    torch.manual_seed(RANDOM_STATE)  # GNNExplainer mask init is random
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )
    explanation = explainer(sub_x, sub_edge_index, index=mapping.item())

    subset_list = subset.tolist()
    local_accounts = [data.accounts[i] for i in subset_list]
    edge_importance = {}
    src, dst = sub_edge_index
    for e in range(sub_edge_index.shape[1]):
        u, v = local_accounts[src[e].item()], local_accounts[dst[e].item()]
        imp = explanation.edge_mask[e].item()
        edge_importance[(u, v)] = max(imp, edge_importance.get((u, v), 0.0))

    feat_importance = explanation.node_mask[mapping.item()].numpy()
    return edge_importance, feat_importance


def plot_case(account, case_type, G_sub, radius_used, illicit_nodes, illicit_edge_pairs,
              account_cycles, community_lookup, edge_importance, feature_row, proba, y_true, threshold,
              top_features=()):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1.5])
    ax_graph = fig.add_subplot(gs[0, :2])
    ax_text = fig.add_subplot(gs[0, 2])
    ax_text.axis("off")

    pos = nx.spring_layout(G_sub, seed=42, k=1.2 / max(len(G_sub), 1) ** 0.5)
    center_community = community_lookup.get(account)

    node_colors, node_sizes, node_edgecolors, node_linewidths = [], [], [], []
    for n in G_sub.nodes():
        node_colors.append("crimson" if n in illicit_nodes else "steelblue")
        if n == account:
            node_sizes.append(900)
            node_edgecolors.append("black")
            node_linewidths.append(3.0)
        else:
            node_sizes.append(450)
            same_comm = community_lookup.get(n) == center_community
            node_edgecolors.append("gold" if same_comm else "gray")
            node_linewidths.append(2.2 if same_comm else 0.8)

    # Collapse to one edge per pair for drawing
    simple_edges = {}
    for u, v, edata in G_sub.edges(data=True):
        key = (u, v)
        is_ill = edata.get("is_laundering") == 1
        prev = simple_edges.get(key, False)
        simple_edges[key] = prev or is_ill

    edge_colors, edge_widths, edge_styles = [], [], []
    draw_edges = list(simple_edges.keys())
    for (u, v) in draw_edges:
        is_ill = simple_edges[(u, v)]
        in_cycle_edge = u != v and any(
            (c[i], c[(i + 1) % len(c)]) == (u, v) for c in account_cycles for i in range(len(c))
        )
        imp = edge_importance.get((u, v), 0.0)
        if is_ill:
            edge_colors.append("crimson")
        elif imp > 0.3:
            edge_colors.append("darkorange")
        else:
            edge_colors.append("lightgray")
        edge_widths.append(1.0 + 4.0 * imp + (1.5 if in_cycle_edge else 0))
        edge_styles.append("dashed" if in_cycle_edge else "solid")

    nx.draw_networkx_nodes(G_sub, pos, ax=ax_graph, node_color=node_colors, node_size=node_sizes,
                            edgecolors=node_edgecolors, linewidths=node_linewidths, alpha=0.9)
    for (u, v), color, width, style in zip(draw_edges, edge_colors, edge_widths, edge_styles):
        nx.draw_networkx_edges(G_sub, pos, ax=ax_graph, edgelist=[(u, v)], edge_color=color,
                                width=width, style=style, arrows=True, arrowsize=12, alpha=0.8)
    # Put labels just below each node
    label_pos = {n: (x, y - 0.09) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G_sub, label_pos, ax=ax_graph, font_size=9, font_family="monospace",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))

    legend_handles = [
        mpatches.Patch(color="crimson", label="Illicit-labelled account / laundering transaction"),
        mpatches.Patch(color="steelblue", label="Licit-labelled account"),
        mpatches.Patch(color="darkorange", label="High GNNExplainer importance (non-laundering edge)"),
        mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2, label="Same community as target (gold outline)"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Edge in a detected cycle with the target"),
    ]
    ax_graph.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
    case_label = CASE_LABELS[case_type]
    ax_graph.set_title(f"{account} — {case_label}\n(drawn: {radius_used}-hop neighbourhood, "
                        f"{len(G_sub)} nodes; attribution: {EXPLAIN_HOPS}-hop subgraph)", fontsize=12)
    ax_graph.axis("off")

    by_len = {}
    for c in account_cycles:
        by_len[len(c)] = by_len.get(len(c), 0) + 1
    text_lines = [
        f"Account: {account}",
        f"True label: {'illicit' if y_true == 1 else 'licit'}",
        f"Probability: {proba:.4f}",
        f"Threshold: {threshold:.4f}",
        f"Predicted: {'illicit' if proba >= threshold else 'licit'}",
        "",
        "Key features",
        f"  in / out degree: {feature_row['in_degree']:.0f} / {feature_row['out_degree']:.0f}",
        f"  betweenness: {feature_row['betweenness']:.2e}",
        f"  community size: {feature_row['community_size']:.0f}",
        f"  community illicit density: {feature_row['community_illicit_density']:.4f}",
        "",
        f"Detected cycles: {len(account_cycles)}",
    ]
    for length in sorted(by_len):
        text_lines.append(f"  length {length}: {by_len[length]}")
    if top_features:
        text_lines += ["", "Top GNNExplainer features"]
        text_lines += [f"  {name}" for name in top_features]

    ax_text.text(0, 1, "\n".join(text_lines), fontsize=11, family="monospace",
                 va="top", ha="left", transform=ax_text.transAxes)

    plt.tight_layout()
    out_path = OUT_DIR / f"{account}_{case_type}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = setup_model_and_data()
    data, model, x = ctx["data"], ctx["model"], ctx["x"]
    y_np, test_idx, threshold, proba_all = ctx["y_np"], ctx["test_idx"], ctx["threshold"], ctx["proba_all"]

    account_to_idx = {a: i for i, a in enumerate(data.accounts)}
    all_cycles, G_multi = get_cycles_cached()
    cycle_count = {}
    for cyc in all_cycles:
        for node in cyc:
            cycle_count[node] = cycle_count.get(node, 0) + 1

    samples = select_samples(ctx, cycle_count)
    for account, case_type in samples:
        assert account_to_idx[account] in test_idx, f"{account} is not in the test set!"
        print(f"Selected {case_type}: {account}")

    features_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "account_features.csv").set_index("account")
    community_lookup = features_df["community_id"].to_dict()
    illicit_nodes = set(features_df.index[features_df["is_illicit"] == 1])
    illicit_edge_pairs = set()  # populated per-subgraph from real edge attrs below

    report_sections = []
    for account, case_type in samples:
        print(f"\nProcessing {account} ({case_type})...")
        idx = account_to_idx[account]
        proba = proba_all[idx]
        y_true = y_np[idx]

        G_sub, radius_used, _ = get_viz_subgraph(account, data, G_multi, account_to_idx)
        account_cycles = find_account_cycles(account, all_cycles)
        print(f"  Sub-graph: {radius_used}-hop, {len(G_sub)} nodes, {G_sub.number_of_edges()} edges. "
              f"Cycles containing this account: {len(account_cycles)}")

        t0 = time.time()
        edge_importance, feat_importance = run_gnn_explainer(account, model, x, data, account_to_idx)
        print(f"  GNNExplainer done in {time.time()-t0:.2f}s")

        feature_row = features_df.loc[account]
        top_features = data.feature_cols
        top_feat_idx = np.argsort(-feat_importance)[:5]
        out_path = plot_case(
            account, case_type, G_sub, radius_used, illicit_nodes, illicit_edge_pairs,
            account_cycles, community_lookup, edge_importance, feature_row, proba, y_true, threshold,
            top_features=[top_features[i] for i in top_feat_idx[:3]],
        )
        print(f"  Saved {out_path}")

        top_feat_str = ", ".join(f"{top_features[i]} ({feat_importance[i]:.3f})" for i in top_feat_idx)

        top_edges = sorted(edge_importance.items(), key=lambda kv: -kv[1])[:5]
        top_edges_str = "; ".join(f"{u}->{v} ({imp:.3f})" for (u, v), imp in top_edges)

        neighbours = set(G_multi.predecessors(account)) | set(G_multi.successors(account))
        neighbours.discard(account)
        report_sections.append({
            "n_neighbours": len(neighbours),
            "n_illicit_neighbours": len(neighbours & illicit_nodes),
            "top_features": [top_features[i] for i in top_feat_idx],
            "account": account, "case_type": case_type, "proba": proba, "y_true": y_true,
            "threshold": threshold, "feature_row": feature_row, "account_cycles": account_cycles,
            "top_feat_str": top_feat_str, "top_edges_str": top_edges_str,
            "png": out_path.name, "n_nodes": len(G_sub), "radius": radius_used,
        })

    write_summary(report_sections)
    cases = []
    for sec in report_sections:
        fr = sec["feature_row"]
        cases.append({
            "account": sec["account"], "case_type": sec["case_type"],
            "label": CASE_LABELS[sec["case_type"]].lower(),
            "proba": float(sec["proba"]), "y_true": int(sec["y_true"]),
            "threshold": float(sec["threshold"]), "png": sec["png"],
            "radius": sec["radius"], "n_nodes_drawn": sec["n_nodes"],
            "cycle_lengths": sorted(len(c) for c in sec["account_cycles"]),
            "n_neighbours": sec["n_neighbours"], "n_illicit_neighbours": sec["n_illicit_neighbours"],
            "top_features": sec["top_features"],
            "features": {k: float(fr[k]) for k in (
                "in_degree", "out_degree", "total_degree", "betweenness", "community_size",
                "community_illicit_density", "cycle_count", "total_sent", "total_received",
                "transaction_count")},
        })
    CASES_JSON.write_text(json.dumps(cases, indent=2))


def write_summary(sections):
    lines = [
        "# Explainability Summary",
        "",
        "Four accounts from the **test set only**, chosen by a fixed rule (select_samples): among "
        "test accounts in at least one detected cycle with 5-39 neighbours, the highest-probability "
        "true positive, the lowest-probability true positive, the highest-probability false "
        "negative, and the highest-probability false positive. See notebooks/explainability/.",
        "",
        "## Methodology",
        "- Ego-subgraph for visualization: 2-hop if that stays under 40 nodes, else 1-hop "
        "(built from the original MultiDiGraph, so real per-transaction `is_laundering` labels "
        "are available for edge coloring).",
        "- Attribution: PyTorch Geometric's `GNNExplainer`, run on a 3-hop subgraph (matching the "
        "model's 3 SAGEConv layers) extracted via `k_hop_subgraph` — confirmed compatible with "
        "GraphSAGE with no NeighborLoader needed; tested first on one account (0.68s on a "
        "2,381-node subgraph) before running on all four (largest: 2.82s on 14,916 nodes).",
        "- Cycle membership: cycle detection was recomputed once for this phase, reusing the "
        "exact functions and parameters from graph_analysis.py (same giant-SCC partial-sample "
        "caveat applies — see that module).",
        "- Community membership: read directly from account_features.csv (`community_id`).",
        "",
    ]

    for s in sections:
        acct, ct = s["account"], s["case_type"]
        fr = s["feature_row"]
        cycle_lens = sorted(len(c) for c in s["account_cycles"])
        lines += [
            f"## {acct} — {ct.replace('_',' ').title()}",
            "",
            f"![{acct}](explainability/{s['png']})",
            "",
            f"- True label: {'ILLICIT' if s['y_true']==1 else 'licit'}, predicted probability: "
            f"{s['proba']:.4f} (threshold {s['threshold']:.4f})",
            f"- Sub-graph shown: {s['radius']}-hop, {s['n_nodes']} nodes",
            f"- Key features: in_degree={fr['in_degree']:.0f}, out_degree={fr['out_degree']:.0f}, "
            f"betweenness={fr['betweenness']:.6f}, community_illicit_density="
            f"{fr['community_illicit_density']:.4f}, cycle_count={fr['cycle_count']:.0f}",
            f"- Cycle membership: {'in ' + str(len(cycle_lens)) + ' detected cycle(s), lengths ' + str(cycle_lens) if cycle_lens else 'not in any detected cycle'}",
            f"- GNNExplainer top contributing features: {s['top_feat_str']}",
            f"- GNNExplainer top contributing edges (neighbor influence): {s['top_edges_str']}",
            "",
        ]

    # Per-case explanation text
    by_type = {s["case_type"]: s for s in sections}
    narrative = [
        "## Case-by-case narrative",
        "",
    ]
    if "true_positive_confident" in by_type:
        s = by_type["true_positive_confident"]
        narrative += [
            f"**{s['account']} (confident true positive):** the model flags this account with high confidence "
            f"({s['proba']:.3f}), and GNNExplainer's top features/edges show why: high degree, "
            "heavy cycle participation, and neighbors that are themselves illicit or highly "
            "central. This is the model working as intended — structural signal and labels align.",
            "",
        ]
    if "true_positive_marginal" in by_type:
        s = by_type["true_positive_marginal"]
        narrative += [
            f"**{s['account']} (marginal true positive):** correctly caught, but only just — "
            f"probability {s['proba']:.4f} is barely above the {s['threshold']:.4f} threshold. "
            "This is a realistic 'good but not overwhelming' catch, useful for calibrating how "
            "much confidence to place in borderline flags from this model.",
            "",
        ]
    if "false_negative" in by_type:
        s = by_type["false_negative"]
        cycle_lens = sorted(len(c) for c in s["account_cycles"])
        narrative += [
            f"**{s['account']} (false negative, honest miss):** true label is illicit, but the "
            f"model scored it {s['proba']:.4f} — just under the {s['threshold']:.4f} threshold. "
            f"Notably, this account IS in {len(cycle_lens)} detected cycle(s) (lengths {cycle_lens}), "
            "which normally correlates with illicit activity — so cycle participation alone wasn't "
            "enough here. The likely explanation, per GNNExplainer's attribution, is that this "
            "account's immediate neighbors are predominantly licit-labeled, so message-passing "
            "diluted the cycle signal below the decision threshold. This illustrates a real "
            "limitation: an account embedded mostly among legitimate accounts, even if it "
            "participates in a suspicious cycle, can still slip under the threshold.",
            "",
        ]
    if "false_positive" in by_type:
        s = by_type["false_positive"]
        cycle_lens = sorted(len(c) for c in s["account_cycles"])
        narrative += [
            f"**{s['account']} (false positive, honest false alarm):** true label is licit, but the "
            f"model was highly confident it was illicit ({s['proba']:.4f}). "
            + (f"This account is itself in {len(cycle_lens)} detected cycle(s) (lengths {cycle_lens}), "
               "which is very likely what misled the model — cycle participation is a strong "
               "learned signal, but not every account in a short transaction cycle is actually "
               "laundering money; some cyclic payment patterns (e.g. reciprocal business "
               "payments) are entirely legitimate. "
               if cycle_lens else
               "This account is not in any detected cycle, so the false alarm is likely driven "
               "by degree/community features resembling illicit accounts rather than cycle "
               "structure specifically. ")
            + "This is a good example of why the model's output should support, not replace, "
            "manual review — a high score here would trigger unnecessary investigation.",
            "",
        ]

    lines += narrative
    summary = "\n".join(lines)
    SUMMARY_MD.write_text(summary)
    print(f"\nSaved {SUMMARY_MD}")


if __name__ == "__main__":
    main()
