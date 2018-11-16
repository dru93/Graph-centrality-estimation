# Graph centrality estimation
Sassy experiments estimating **closeness** and **betweenness** centrality measures based on pivot selection strategies from *Centrality Estimation in Large Networks* 2006 paper.

link to paper: http://algo.uni-konstanz.de/publications/bp-celn-06.pdf

### Pivot selection strategies

| Strategy | Selection rule                               |
|:--------:|:--------------------------------------------:|
| `Random` | uniformly at random                          |
| `RanDeg` | random proportional to degree node value     |
| `MaxMin` | maximize minimum distance to previous pivots |
| `MaxSum` | maximize sum of distances to previous pivots |
| `MinSum` | minimize sum of distances to previous pivots |
| `Mixed3` | alternate between MaxMin, MaxSum, and Random |
| `pgRank` | random proportional to page rank node value  |

### Graphs

| Graph                   | Number of nodes | Notes                    |
|:-----------------------:|:---------------:|:------------------------:|
| `Erdős-Rényi`           | 1000            | p = 0.5                  |
| `Watts-Strogatz`        | 1000            | p = 0.5, k =10           |
| `Barabási-Albert`       | 1000            | m = nodes/2      |
| `Online social network` | 2862 or 279630  | courtesy of assignment 1 |
| `Twitter mention graph` | ?               | courtesy of assignment 2 |

Note: number of nodes refers to numbers of nodes of largest connected component

Graphs must be:
1. undirected
2. connected
3. unweighted

To do:
* find best pivot selection strategy:
* find pivot means of centrality & betweenness values for each strategy
* compute one-sample t-test between pivot means and exact values