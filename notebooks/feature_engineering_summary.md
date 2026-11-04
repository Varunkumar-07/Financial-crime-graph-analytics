# Feature Engineering Summary

## Methodology notes
- Betweenness centrality: APPROXIMATE, k=500 random-source sampling (recomputed with the same parameters as graph_analysis.py).
- Louvain communities: recomputed with seed=42 on the undirected, transaction-count-weighted graph.
- community_illicit_density: computed from TRAINING-split labels only, leave-one-out (each account's own label excluded); communities with no other training members use the global training illicit rate. No validation/test label enters any feature.
- Cycle participation (length 2-6): 10,651 total cycles (shared cache) — exact for 2/3-cycles and small-SCC 4-6 cycles, PARTIAL SAMPLE (120s cap) for the giant 17,075-node SCC. cycle_count is therefore a lower bound for accounts in that component.
- Observed dataset window: 17.679 days (computed directly from min/max Timestamp). A `velocity` feature (transaction_count / window) was considered but dropped: since the window is constant across all accounts, it's a pure linear rescaling of `transaction_count` — perfectly collinear, adds no information, and was found to distort tree-based feature importance rankings (see baseline_model_summary.md).

## Feature table shape
- 515,080 rows x 16 columns

## Class balance (is_illicit)
- Licit (0): 508,723 (98.7658%)
- Illicit (1): 6,357 (1.2342%)

## Sanity checks
### Missing values (NaN)
- `in_out_amount_ratio`: 18,085 (expected: accounts with zero total_sent have an undefined in_out_amount_ratio)
### Negative-value checks
- total_sent < 0: 0
- total_received < 0: 0
- in_degree < 0: 0
- out_degree < 0: 0
### Infinite-value checks
- None

## Correlation with is_illicit (sorted by |correlation|)

| Feature | Correlation | Licit mean | Illicit mean |
|---|---|---|---|
| in_degree | 0.0974372 | 1.94334 | 4.26538 |
| cycle_flag | 0.0918857 | 0.0119849 | 0.106811 |
| cycle_count | 0.0763353 | 0.0501314 | 0.457291 |
| community_illicit_density | 0.0749028 | 0.0146768 | 0.0240324 |
| betweenness | 0.0621372 | 1.95239e-06 | 2.00477e-05 |
| total_degree | 0.0408041 | 3.82884 | 13.159 |
| transaction_count | 0.0322835 | 18.6811 | 102.751 |
| out_degree | 0.0319764 | 1.8855 | 8.89366 |
| community_size | 0.0274605 | 16140.2 | 21811.7 |
| total_sent | 0.00495409 | 4.2285e+07 | 2.1839e+08 |
| avg_transaction_amount | 0.00245508 | 4.26042e+06 | 1.5196e+07 |
| in_out_amount_ratio | -0.000526357 | 19019 | 3262.05 |
| total_received | 0.000208138 | 5.893e+07 | 6.82332e+07 |

Total runtime: 738.0s
