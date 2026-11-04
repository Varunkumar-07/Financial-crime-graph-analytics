# GNN Baseline Summary (GraphSAGE)

## Setup
- Graph: 515,080 nodes, 1,656,289 directed edges (collapsed multi-edges, made undirected — see build_pyg_data.py)
- Node features: 13 (same as classical baselines, excluding account/community_id/is_illicit)
- Feature preprocessing: log1p on heavily right-skewed count/dollar columns, then z-score standardization using TRAIN-set statistics only. Required: an initial run on raw (unscaled) features — fine for the tree-based baselines, which are scale-invariant — produced loss values in the tens of millions and test AUC-ROC 0.5034 (no better than random) for GraphSAGE, since features like total_sent range into the hundreds of millions and destabilized training.
- Architecture: 3-layer GraphSAGE, hidden_dim=64, dropout=0.3, ReLU
- Split: shared 70/10/20 train/validation/test split (splits.py, stratified), identical to the classical baselines; validation drives early stopping and threshold selection — test set is never used for anything but final reported metrics
- Class imbalance handling: weighted cross-entropy loss, illicit class weight=80.02 (train neg/pos ratio) — the GNN analogue of scale_pos_weight/class_weight='balanced' in the classical baselines
- Optimizer: Adam, lr=0.01, weight_decay=0.0005
- Trained 200 epochs (early stopping, patience=20) in 276.8s (1.384s/epoch) on CPU
- Best validation AUC-PR: 0.2682

## Test set evaluation
- AUC-ROC: 0.9179
- AUC-PR (average precision): 0.2470

**At default threshold 0.5:** precision=0.0531, recall=0.8631, f1=0.1000

**At chosen threshold 0.8164 (F2-optimal on validation set, not test — avoids the threshold-leakage caveat noted in the classical baseline report):** precision=0.1639, recall=0.4784, f1=0.2442

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 98,644 (TN) | 3,101 (FP) |
| Actual illicit | 663 (FN) | 608 (TP) |

## Comparison to classical baselines (same test set)

| Model | AUC-ROC | AUC-PR |
|---|---|---|
| XGBoost | 0.8534 | 0.1229 |
| Random Forest | 0.8470 | 0.0864 |
| GraphSAGE | 0.9179 | 0.2470 |

Total runtime: 277.6s
