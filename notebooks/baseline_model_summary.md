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

- AUC-ROC: 0.8545
- AUC-PR (average precision): 0.1276

**At default threshold 0.5:** precision=0.0485, recall=0.6664, f1=0.0904

**At chosen threshold 0.7355 (maximizes F2 on the validation set — recall weighted 4x more than precision):** precision=0.0954, recall=0.3910, f1=0.1533 (validation F2=0.2411)

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 97,031 (TN) | 4,714 (FP) |
| Actual illicit | 774 (FN) | 497 (TP) |

### Random Forest

- AUC-ROC: 0.8470
- AUC-PR (average precision): 0.0858

**At default threshold 0.5:** precision=0.0503, recall=0.5964, f1=0.0927

**At chosen threshold 0.5894 (maximizes F2 on the validation set — recall weighted 4x more than precision):** precision=0.0633, recall=0.4524, f1=0.1110 (validation F2=0.2097)

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 93,230 (TN) | 8,515 (FP) |
| Actual illicit | 696 (FN) | 575 (TP) |

## Feature importance

XGBoost importance = gain-based, normalized to sum to 1. Random Forest importance = mean decrease in impurity. `avg_rank` = average of each model's rank (1=most important).

| Feature | XGB gain (norm.) | RF importance | Avg rank |
|---|---|---|---|
| total_degree | 0.4346 | 0.1468 | 1.0 |
| community_size | 0.0775 | 0.1034 | 2.5 |
| in_degree | 0.0559 | 0.1013 | 4.5 |
| transaction_count | 0.0480 | 0.1339 | 4.5 |
| out_degree | 0.0597 | 0.0716 | 6.5 |
| total_received | 0.0444 | 0.0904 | 7.0 |
| in_out_amount_ratio | 0.0471 | 0.0881 | 7.0 |
| cycle_count | 0.0756 | 0.0100 | 8.0 |
| betweenness | 0.0557 | 0.0199 | 8.5 |
| community_illicit_density | 0.0394 | 0.0856 | 8.5 |
| avg_transaction_amount | 0.0295 | 0.0717 | 10.0 |
| total_sent | 0.0326 | 0.0670 | 10.5 |
| cycle_flag | 0.0000 | 0.0103 | 12.5 |

## Comparison to Phase 6 correlation findings
- Top 5 features by average model rank: ['total_degree', 'community_size', 'in_degree', 'transaction_count', 'out_degree']
- Phase 6 flagged `community_illicit_density` and `cycle_count` as promising nonlinear signals (weak linear correlation but large mean gap between classes).
- Neither flagged feature reached the top 5 in these models — see full importance table above for their actual rank.

Total runtime: 18.1s
