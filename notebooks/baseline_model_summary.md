# Baseline Model Summary

## Setup
- Feature table: 515,080 rows, 13 features (excluded: ['account', 'community_id'])
- Split: 80/20 train/test, stratified on `is_illicit`, random_state=42
- Train set: 412,064 rows (1.2343% positive)
- Test set: 103,016 rows (1.2338% positive)
- Class imbalance handling: class weighting only (no SMOTE/oversampling). XGBoost `scale_pos_weight`=80.02 (train neg/pos ratio); Random Forest `class_weight='balanced'`.
- `in_out_amount_ratio` NaNs (accounts that never sent) imputed with -1 (an impossible value for the true ratio, so tree splits isolate this group cleanly).
- `community_id` excluded from features (arbitrary high-cardinality label; `community_illicit_density` and `community_size` are the informative derivatives, per Phase 6).

## Threshold selection rationale
Rather than the default 0.5 cutoff, the reported "chosen" threshold for each model maximizes the F2-score (recall weighted 4x more heavily than precision) on the test set's precision-recall curve. This reflects that in an AML screening context, missing an illicit account (false negative) is typically costlier than an extra manual review (false positive). Both the default-0.5 and F2-optimal metrics are reported below for comparison. Note: for a production pipeline the threshold should be tuned on a held-out validation split rather than the test set itself to avoid threshold-selection leakage — this baseline uses the test set directly for simplicity and should be revisited.

## Evaluation

### XGBoost

- AUC-ROC: 0.8658
- AUC-PR (average precision): 0.1459

**At default threshold 0.5:** precision=0.0499, recall=0.6971, f1=0.0931

**At chosen threshold 0.7515 (maximizes F2 — weights recall 4x more than precision, since missing an illicit account is costlier than a false positive here):** precision=0.1084, recall=0.4052, f1=0.1710, f2=0.2618

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 97,509 (TN) | 4,236 (FP) |
| Actual illicit | 756 (FN) | 515 (TP) |

### Random Forest

- AUC-ROC: 0.8561
- AUC-PR (average precision): 0.0976

**At default threshold 0.5:** precision=0.0497, recall=0.6483, f1=0.0923

**At chosen threshold 0.6886 (maximizes F2 — weights recall 4x more than precision, since missing an illicit account is costlier than a false positive here):** precision=0.0923, recall=0.3525, f1=0.1463, f2=0.2254

Confusion matrix at chosen threshold:

| | Predicted licit | Predicted illicit |
|---|---|---|
| Actual licit | 97,338 (TN) | 4,407 (FP) |
| Actual illicit | 823 (FN) | 448 (TP) |

## Feature importance

XGBoost importance = gain-based, normalized to sum to 1. Random Forest importance = mean decrease in impurity. `avg_rank` = average of each model's rank (1=most important).

| Feature | XGB gain (norm.) | RF importance | Avg rank |
|---|---|---|---|
| total_degree | 0.2744 | 0.1436 | 1.5 |
| community_illicit_density | 0.2211 | 0.1805 | 1.5 |
| in_degree | 0.0580 | 0.0914 | 5.0 |
| total_received | 0.0653 | 0.0899 | 5.0 |
| transaction_count | 0.0497 | 0.1156 | 5.5 |
| out_degree | 0.0675 | 0.0568 | 7.0 |
| cycle_count | 0.0793 | 0.0108 | 7.5 |
| in_out_amount_ratio | 0.0428 | 0.0794 | 8.0 |
| betweenness | 0.0577 | 0.0189 | 9.0 |
| community_size | 0.0256 | 0.0809 | 9.0 |
| total_sent | 0.0305 | 0.0583 | 9.5 |
| avg_transaction_amount | 0.0280 | 0.0642 | 9.5 |
| cycle_flag | 0.0000 | 0.0098 | 13.0 |

## Comparison to Phase 6 correlation findings
- Top 5 features by average model rank: ['total_degree', 'community_illicit_density', 'in_degree', 'total_received', 'transaction_count']
- Phase 6 flagged `community_illicit_density` and `cycle_count` as promising nonlinear signals (weak linear correlation but large mean gap between classes).
- **Confirmed**: ['community_illicit_density'] rank in the top 5 for at least one model — the tree-based models did pick up on this nonlinear signal that raw correlation understated.

Total runtime: 11.6s
