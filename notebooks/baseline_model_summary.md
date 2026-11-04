# Baseline Model Summary

## Setup
- Feature table: 515,080 rows, 13 features (excluded: ['account', 'community_id'])
- Split: shared 70/10/20 train/validation/test split from splits.py (stratified on `is_illicit`, random_state=42) — identical to the GraphSAGE split
- Train set: 360,556 rows (1.2342% positive)
- Validation set: 51,508 rows (1.2348% positive)
- Test set: 103,016 rows (1.2338% positive)
- Class imbalance handling: class weighting only (no SMOTE/oversampling). XGBoost `scale_pos_weight`=80.02 (train neg/pos ratio); Random Forest `class_weight='balanced'`.
- `in_out_amount_ratio` NaNs (accounts that never sent) imputed with -1 (an impossible value for the true ratio, so tree splits isolate this group cleanly).
- `community_id` excluded from features (arbitrary high-cardinality label; `community_illicit_density` and `community_size` are the informative derivatives, per Phase 6).

## Threshold selection rationale
Rather than the default 0.5 cutoff, the reported "chosen" threshold for each model maximizes the F2-score (recall weighted 4x more heavily than precision) on the VALIDATION set's precision-recall curve, and is then applied unchanged to the test set — the same protocol as GraphSAGE, so no test label influences any reported threshold. Both the default-0.5 and F2-optimal metrics are reported below.

## Evaluation

### XGBoost

- AUC-ROC: 0.8534
- AUC-PR (average precision): 0.1229

**At default threshold 0.5:** precision=0.0476, recall=0.6577, f1=0.0887

**At chosen threshold 0.7352 (maximizes F2 on the validation set — recall weighted 4x more than precision):** precision=0.0918, recall=0.3698, f1=0.1470 (validation F2=0.2479)

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 97,093 (TN) | 4,652 (FP) |
| Actual illicit | 801 (FN) | 470 (TP) |

### Random Forest

- AUC-ROC: 0.8470
- AUC-PR (average precision): 0.0864

**At default threshold 0.5:** precision=0.0509, recall=0.5987, f1=0.0939

**At chosen threshold 0.6070 (maximizes F2 on the validation set — recall weighted 4x more than precision):** precision=0.0673, recall=0.4186, f1=0.1159 (validation F2=0.2077)

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 94,371 (TN) | 7,374 (FP) |
| Actual illicit | 739 (FN) | 532 (TP) |

## Feature importance

XGBoost importance = gain-based, normalized to sum to 1. Random Forest importance = mean decrease in impurity. `avg_rank` = average of each model's rank (1=most important).

| Feature | XGB gain (norm.) | RF importance | Avg rank |
|---|---|---|---|
| total_degree | 0.4207 | 0.1470 | 1.0 |
| community_size | 0.0770 | 0.1035 | 3.0 |
| transaction_count | 0.0486 | 0.1335 | 4.5 |
| in_degree | 0.0590 | 0.1009 | 5.0 |
| out_degree | 0.0613 | 0.0707 | 6.5 |
| cycle_count | 0.0774 | 0.0105 | 7.0 |
| total_received | 0.0464 | 0.0912 | 7.0 |
| in_out_amount_ratio | 0.0464 | 0.0884 | 7.0 |
| betweenness | 0.0603 | 0.0204 | 8.0 |
| community_illicit_density | 0.0393 | 0.0857 | 8.5 |
| avg_transaction_amount | 0.0306 | 0.0724 | 10.0 |
| total_sent | 0.0331 | 0.0666 | 10.5 |
| cycle_flag | 0.0000 | 0.0093 | 13.0 |

## Comparison to Phase 6 correlation findings
- Top 5 features by average model rank: ['total_degree', 'community_size', 'transaction_count', 'in_degree', 'out_degree']
- Phase 6 flagged `community_illicit_density` and `cycle_count` as promising nonlinear signals (weak linear correlation but large mean gap between classes).
- Neither flagged feature reached the top 5 in these models — see full importance table above for their actual rank.

Total runtime: 11.0s
