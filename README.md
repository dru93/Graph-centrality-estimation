# Graph centrality estimation
Sassy experiments estimating **closeness** and **betweenness** centrality measures based on pivot selection strategies from *Centrality Estimation in Large Networks* paper.

link to paper: `http://algo.uni-konstanz.de/publications/bp-celn-06.pdf`

## Pivot selection strategies

| Strategy | Selection rule                               |
|:--------:|:--------------------------------------------:|
| `Random` | uniformly at random                          |
| `RanDeg` | random proportional to degree node value     |
| `pgRank` | random proportional to page rank node value  |
| `MaxMin` | maximize minimum distance to previous pivots |
| `MaxSum` | maximize sum of distances to previous pivots |
| `MinSum` | minimize sum of distances to previous pivots |
| `Mixed3` | alternate between MaxMin, MaxSum, and Random |

## Graphs

| Graph                   | Number of nodes |
|:-----------------------:|:---------------:|
| `Erdős-Rényi`           | 1000            |
| `US power grid`         | 4941            |
| `Price network  `       | 100000          |
| `Online social network` | 2862 or 279630  |
| `Twitter mention graph` | ?               |