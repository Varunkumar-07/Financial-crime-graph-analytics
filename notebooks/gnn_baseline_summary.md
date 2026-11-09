# GNN Baseline Summary (GraphSAGE)

## Setup
- Graph: 515,080 nodes, 1,656,289 directed edges (collapsed multi-edges, made undirected — see build_pyg_data.py)
- Node features: 13 (same as classical baselines, excluding account/community_id/is_illicit)
- Feature preprocessing: log1p on heavily right-skewed count/dollar columns, then z-score standardization using TRAIN-set statistics only. Required: an initial run on raw (unscaled) features — fine for the tree-based baselines, which are scale-invariant — produced loss values in the tens of millions and test AUC-ROC 0.5034 (no better than random) for GraphSAGE, since features like total_sent range into the hundreds of millions and destabilized training.
- Architecture: 3-layer GraphSAGE, hidden_dim=64, dropout=0.3, ReLU
- Split: shared 70/10/20 train/validation/test split (splits.py, stratified), identical to the classical baselines; validation drives early stopping and threshold selection — test set is never used for anything but final reported metrics
- Class imbalance handling: weighted cross-entropy loss, illicit class weight=80.02 (train neg/pos ratio) — the GNN analogue of scale_pos_weight/class_weight='balanced' in the classical baselines
- Optimizer: Adam, lr=0.01, weight_decay=0.0005
- Trained 163 epochs (early stopping, patience=20) in 328.6s (2.016s/epoch) on CPU
- Best validation AUC-PR: 0.2517

## Test set evaluation
- AUC-ROC: 0.9143
- AUC-PR (average precision): 0.2325

**At default threshold 0.5:** precision=0.0523, recall=0.8592, f1=0.0986

**At chosen threshold 0.7924 (F2-optimal on validation set, not test — avoids the threshold-leakage caveat noted in the classical baseline report):** precision=0.1403, recall=0.5075, f1=0.2198

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 97,792 (TN) | 3,953 (FP) |
| Actual illicit | 626 (FN) | 645 (TP) |

## Comparison to classical baselines (same test set)

| Model | AUC-ROC | AUC-PR |
|---|---|---|
| XGBoost | 0.8545 | 0.1276 |
| Random Forest | 0.8470 | 0.0858 |
| GraphSAGE | 0.9143 | 0.2325 |

Total runtime: 330.0s
