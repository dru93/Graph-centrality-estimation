# Graph centrality estimation
Sassy experiments estimating **closeness** and **betweenness** centrality measures based on pivot selection strategies from *[Centrality Estimation in Large Networks]*  paper.

[Centrality Estimation in Large Networks]: http://algo.uni-konstanz.de/publications/bp-celn-06.pdf

## Pivot selection strategies

| Strategy    | Selection rule                               |
|:-----------:|:--------------------------------------------:|
| `Random`    | uniformly at random                          |
| `RanDeg`    | random proportional to degree node value     |
| `RanPgRank` | random proportional to page rank node value  |
| `Degree`    | maximize degree node value                   |
| `pgRank`    | maximize page rank node value                |
| `pgRankRev` | minimize page rank node value                |
| `pgRankAlt` | alternate `pgRank` & `pgRankRev`             |
| `MaxMin`    | maximize minimum distance to previous pivots |
| `MaxSum`    | maximize sum of distances to previous pivots |
| `MinSum`    | minimize sum of distances to previous pivots |
| `Mixed3`    | alternate `MaxMin`, `MaxSum`, and `MinSum`   |

## Graphs

| Graph                   | Number of nodes |
|:-----------------------:|:---------------:|
| `Erdős-Rényi`           | $10^5$          |
| `Barabási–Albert`       | $10^5$          |
| `Price network `        | $10^5$          |
| `US power grid`         | $4941$          |
| `Online social network` | $279630$        |