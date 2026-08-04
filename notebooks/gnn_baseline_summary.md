# GNN Baseline Summary (GraphSAGE)

> Regenerated via an eval-only pass (no retraining) during the Phase 13 integration
> test, after a clean-checkout rebuild test temporarily produced a different trained
> model. This file now reflects the actual model stored at
> `data/processed/graphsage_baseline.pt` — confirmed by loading that exact file and
> re-running evaluation, which reproduced these numbers precisely. See
> `notebooks/integration_test_summary.md` for the full story: GraphSAGE training is
> not bit-reproducible across runs on this machine (multi-threaded CPU floating-point
> non-associativity) even with a fixed random seed, unlike the classical baselines and
> the graph-analysis algorithms, which do reproduce exactly.

## Setup
- Graph: 515,080 nodes, 1,656,289 directed edges (collapsed multi-edges, made undirected — see build_pyg_data.py)
- Node features: 13 (same as classical baselines, excluding account/community_id/is_illicit)
- Feature preprocessing: log1p on heavily right-skewed count/dollar columns, then z-score standardization using TRAIN-set statistics only. Required: an initial run on raw (unscaled) features — fine for the tree-based baselines, which are scale-invariant — produced loss values in the tens of millions and test AUC-ROC 0.5034 (no better than random) for GraphSAGE, since features like total_sent range into the hundreds of millions and destabilized training.
- Architecture: 3-layer GraphSAGE, hidden_dim=64, dropout=0.3, ReLU
- Split: 80/20 train+val / test (stratified), with train further split 90/10 into train/val (stratified) for early stopping and threshold selection — test set is never used for anything but final reported metrics
- Class imbalance handling: weighted cross-entropy loss, illicit class weight=80.02 (train neg/pos ratio) — the GNN analogue of scale_pos_weight/class_weight='balanced' in the classical baselines
- Optimizer: Adam, lr=0.01, weight_decay=0.0005

## Test set evaluation
- AUC-ROC: 0.9218
- AUC-PR (average precision): 0.2740

**At default threshold 0.5:** precision=0.0502, recall=0.8843

**At chosen threshold 0.8727 (F2-optimal on validation set, not test — avoids the threshold-leakage caveat noted in the classical baseline report):** precision=0.1966, recall=0.3973, f1=0.2630

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 99,681 (TN) | 2,064 (FP) |
| Actual illicit | 766 (FN) | 505 (TP) |

## Comparison to classical baselines (Phase 7)

| Model | AUC-ROC | AUC-PR |
|---|---|---|
| XGBoost | 0.8650 | 0.1455 |
| Random Forest | 0.8560 | 0.1019 |
| GraphSAGE | 0.9218 | 0.2740 |
