# Graph Analysis Summary

## Scope and methodology notes
- Betweenness centrality: APPROXIMATE, k=500 random-source sampling (exact is infeasible at 515K nodes).
- Cycle detection (length 2-6): EXACT for 2- and 3-cycles (whole graph) and for 4-6 length cycles within the 2927 small strongly-connected components (max size 12).
- Cycle detection within the giant 17,075-node strongly connected component: exact (exhaustive depth-limited search).

## Betweenness centrality — top 20 accounts
- Of the top 20 accounts by (approximate) betweenness centrality, 4 are labeled illicit (touch at least one illicit transaction).

| Rank | Account | Betweenness (approx) | Illicit? |
|---|---|---|---|
| 1 | 80154AAF0 | 0.002283 | yes |
| 2 | 8001BB380 | 0.001859 | yes |
| 3 | 80010D740 | 0.001739 | yes |
| 4 | 800236DB0 | 0.001721 |  |
| 5 | 800DE74E0 | 0.001697 |  |
| 6 | 8027C2890 | 0.001688 |  |
| 7 | 802EE3990 | 0.001687 |  |
| 8 | 80F60BA50 | 0.001687 |  |
| 9 | 80F60BAF0 | 0.001687 |  |
| 10 | 8013BDF80 | 0.001665 |  |
| 11 | 801B1EFC0 | 0.001662 |  |
| 12 | 80005FD30 | 0.001627 | yes |
| 13 | 8121A7840 | 0.001611 |  |
| 14 | 81219CCD0 | 0.001610 |  |
| 15 | 8000F7850 | 0.001596 |  |
| 16 | 80EC506E0 | 0.001592 |  |
| 17 | 80EC50690 | 0.001592 |  |
| 18 | 807EC03D0 | 0.001583 |  |
| 19 | 807F7B490 | 0.001582 |  |
| 20 | 800081CA0 | 0.001578 |  |

## Community detection (Louvain, undirected; weight 2 if money moved both ways, else 1)
- Number of communities found: 115,975
- Community size distribution — largest: 57,698, median: 1, smallest: 1
- Top 10 community sizes: [57698, 57620, 15618, 14409, 13214, 13108, 12171, 11345, 11125, 10033]
- Illicit accounts fall into 298 distinct communities (out of 6,357 illicit accounts placed in any community).
- 17.57% of illicit accounts fall into the single largest illicit-containing community (community #21899, 1,117 illicit accounts) — the remainder are spread across the other 297 communities.

## Cycle detection (directed cycles, length 2-6)
- 2-cycles (exact, whole graph): 3,693
- 3-cycles (exact, whole graph): 563
- 4-6 length cycles in small SCCs (exact): 119
- 4-6 length cycles in giant SCC (exact): 780
- **Total cycles found: 5,155**
- Cycles involving at least one illicit-labeled edge: 1,080 (20.95% of all cycles found)

## Visualization
- Saved to `notebooks/suspicious_cluster.png`: Detected 6-account cycle containing a laundering transaction

Total analysis runtime: 489.2s
