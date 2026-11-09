# Ablation Summary

How much the reported test scores depend on three methodology choices.
Thresholds are always picked on validation data (max F2).

| Setup | Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| As reported | XGBoost | 0.8534 | 0.1229 | 0.0918 | 0.3698 | 0.1470 |
| As reported | GraphSAGE | 0.9179 | 0.2470 | 0.1639 | 0.4784 | 0.2442 |
| No community_illicit_density | XGBoost | 0.8515 | 0.1077 | 0.0991 | 0.3478 | 0.1542 |
| No community_illicit_density | GraphSAGE | 0.9142 | 0.2350 | 0.1442 | 0.4776 | 0.2215 |
| Time-based split | XGBoost | 0.9191 | 0.1395 | 0.1462 | 0.3126 | 0.1992 |
| Time-based split | GraphSAGE | 0.9493 | 0.2630 | 0.2094 | 0.4529 | 0.2864 |
| Separate early-stop / threshold sets | GraphSAGE | 0.9179 | 0.2470 | 0.1218 | 0.5720 | 0.2008 |

Notes:
- Time-based split: accounts ordered by first transaction; oldest 70% train, next 10% val, newest 20% test (test positive rate 0.55%). community_illicit_density is recomputed from the new training labels; other features still use the full period, so this is a partial check.
- Separate sets: validation split in half, one half for early stopping, the other for the threshold (GraphSAGE only; XGBoost has no early stopping).

Total runtime: 1531s
