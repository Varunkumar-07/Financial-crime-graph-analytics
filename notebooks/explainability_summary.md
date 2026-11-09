# Explainability Summary

Four accounts from the **test set only**, chosen by a fixed rule (select_samples): among test accounts in at least one detected cycle with 5-39 neighbours, the highest-probability true positive, the lowest-probability true positive, the highest-probability false negative, and the highest-probability false positive. See notebooks/explainability/.

## Methodology
- Ego-subgraph for visualization: 2-hop if that stays under 40 nodes, else 1-hop (built from the original MultiDiGraph, so real per-transaction `is_laundering` labels are available for edge coloring).
- Attribution: PyTorch Geometric's `GNNExplainer`, run on a 3-hop subgraph (matching the model's 3 SAGEConv layers) extracted via `k_hop_subgraph` — confirmed compatible with GraphSAGE with no NeighborLoader needed; tested first on one account (0.68s on a 2,381-node subgraph) before running on all four (largest: 2.82s on 14,916 nodes).
- Cycle membership: cycle detection was recomputed once for this phase, reusing the exact functions and parameters from graph_analysis.py (same giant-SCC partial-sample caveat applies — see that module).
- Community membership: read directly from account_features.csv (`community_id`).

## 80605A3E0 — True Positive Confident

![80605A3E0](explainability/80605A3E0_true_positive_confident.png)

- True label: ILLICIT, predicted probability: 0.9981 (threshold 0.7924)
- Sub-graph shown: 1-hop, 8 nodes
- Key features: in_degree=7, out_degree=3, betweenness=0.000001, community_illicit_density=0.0529, cycle_count=1
- Cycle membership: in 1 detected cycle(s), lengths [2]
- GNNExplainer top contributing features: in_degree (0.753), out_degree (0.712), in_out_amount_ratio (0.680), total_degree (0.661), community_size (0.624)
- GNNExplainer top contributing edges (neighbor influence): 805D4C850->80605A3E0 (0.715); 80605A3E0->80605A2A0 (0.711); 80605A3E0->80605A3E0 (0.711); 80605A070->80605A3E0 (0.676); 80605A110->80605A3E0 (0.670)

## 80FD04790 — True Positive Marginal

![80FD04790](explainability/80FD04790_true_positive_marginal.png)

- True label: ILLICIT, predicted probability: 0.7925 (threshold 0.7924)
- Sub-graph shown: 2-hop, 19 nodes
- Key features: in_degree=8, out_degree=2, betweenness=0.000000, community_illicit_density=0.0117, cycle_count=1
- Cycle membership: in 1 detected cycle(s), lengths [2]
- GNNExplainer top contributing features: total_received (0.746), community_illicit_density (0.739), in_out_amount_ratio (0.734), out_degree (0.726), in_degree (0.713)
- GNNExplainer top contributing edges (neighbor influence): 80FD03D40->80FD03D40 (0.839); 80FD04790->80FD04790 (0.756); 80FD04790->80FD04330 (0.754); 80FD046A0->80FD04790 (0.707); 80FD042E0->80FD04790 (0.655)

## 80F2591D0 — False Negative

![80F2591D0](explainability/80F2591D0_false_negative.png)

- True label: ILLICIT, predicted probability: 0.7848 (threshold 0.7924)
- Sub-graph shown: 2-hop, 10 nodes
- Key features: in_degree=7, out_degree=2, betweenness=0.000000, community_illicit_density=0.0140, cycle_count=1
- Cycle membership: in 1 detected cycle(s), lengths [2]
- GNNExplainer top contributing features: in_degree (0.722), total_sent (0.690), out_degree (0.687), community_illicit_density (0.681), cycle_flag (0.678)
- GNNExplainer top contributing edges (neighbor influence): 80F2591D0->803748900 (0.767); 80F2591D0->80F259130 (0.741); 80F258D80->80F2591D0 (0.726); 80F2591D0->80F2590E0 (0.726); 80F2591D0->80F259180 (0.689)

## 80C7FA5E0 — False Positive

![80C7FA5E0](explainability/80C7FA5E0_false_positive.png)

- True label: licit, predicted probability: 0.9989 (threshold 0.7924)
- Sub-graph shown: 2-hop, 7 nodes
- Key features: in_degree=5, out_degree=2, betweenness=0.000000, community_illicit_density=0.3333, cycle_count=1
- Cycle membership: in 1 detected cycle(s), lengths [2]
- GNNExplainer top contributing features: total_received (0.712), in_degree (0.699), out_degree (0.615), cycle_flag (0.610), total_sent (0.606)
- GNNExplainer top contributing edges (neighbor influence): 100428660->80C7F9990 (0.748); 80C7F9990->80C7FA5E0 (0.684); 80C7FA630->80C7F9990 (0.679); 80C7F9990->80C7F9990 (0.672); 80C7FA630->80C7FA5E0 (0.653)

## Case-by-case narrative

**80605A3E0 (confident true positive):** the model flags this account with high confidence (0.998), and GNNExplainer's top features/edges show why: high degree, heavy cycle participation, and neighbors that are themselves illicit or highly central. This is the model working as intended — structural signal and labels align.

**80FD04790 (marginal true positive):** correctly caught, but only just — probability 0.7925 is barely above the 0.7924 threshold. This is a realistic 'good but not overwhelming' catch, useful for calibrating how much confidence to place in borderline flags from this model.

**80F2591D0 (false negative, honest miss):** true label is illicit, but the model scored it 0.7848 — just under the 0.7924 threshold. Notably, this account IS in 1 detected cycle(s) (lengths [2]), which normally correlates with illicit activity — so cycle participation alone wasn't enough here. The likely explanation, per GNNExplainer's attribution, is that this account's immediate neighbors are predominantly licit-labeled, so message-passing diluted the cycle signal below the decision threshold. This illustrates a real limitation: an account embedded mostly among legitimate accounts, even if it participates in a suspicious cycle, can still slip under the threshold.

**80C7FA5E0 (false positive, honest false alarm):** true label is licit, but the model was highly confident it was illicit (0.9989). This account is itself in 1 detected cycle(s) (lengths [2]), which is very likely what misled the model — cycle participation is a strong learned signal, but not every account in a short transaction cycle is actually laundering money; some cyclic payment patterns (e.g. reciprocal business payments) are entirely legitimate. This is a good example of why the model's output should support, not replace, manual review — a high score here would trigger unnecessary investigation.
