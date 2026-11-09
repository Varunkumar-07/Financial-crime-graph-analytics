# Ablation Summary

How much the reported test scores depend on three methodology choices.
Thresholds are always picked on validation data (max F2).

| Setup | Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| As reported | XGBoost | 0.8545 | 0.1276 | 0.0954 | 0.3910 | 0.1533 |
| As reported | GraphSAGE | 0.9143 | 0.2325 | 0.1403 | 0.5075 | 0.2198 |
| No community_illicit_density | XGBoost | 0.8505 | 0.1126 | 0.0902 | 0.4099 | 0.1478 |
| No community_illicit_density | GraphSAGE | 0.9101 | 0.2067 | 0.1279 | 0.5114 | 0.2046 |
| Time-based split | XGBoost | 0.9216 | 0.1280 | 0.1481 | 0.3055 | 0.1995 |
| Time-based split | GraphSAGE | 0.9510 | 0.2573 | 0.2118 | 0.4654 | 0.2911 |
| Separate early-stop / threshold sets | GraphSAGE | 0.9143 | 0.2358 | 0.1448 | 0.4910 | 0.2236 |

Notes:
- Time-based split: accounts ordered by first transaction; oldest 70% train, next 10% val, newest 20% test (test positive rate 0.55%). community_illicit_density is recomputed from the new training labels; other features still use the full period, so this is a partial check.
- Separate sets: validation split in half, one half for early stopping, the other for the threshold (GraphSAGE only; XGBoost has no early stopping).

Total runtime: 1520s
