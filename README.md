# Graph centrality estimation
Sassy experiments estimating **closeness** and **betweenness** centrality measures based on pivot selection strategies from *Centrality Estimation in Large Networks* paper.

link to paper: `http://algo.uni-konstanz.de/publications/bp-celn-06.pdf`

## Pivot selection strategies

| Strategy    | Selection rule                               |
|:-----------:|:--------------------------------------------:|
| `Random`    | uniformly at random                          |
| `Degree`    | maximize degree node value                   |
| `RanDeg`    | random proportional to degree node value     |
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
| `Erdős-Rényi`           | 1000            |
| `US power grid`         | 4941            |
| `Price network  `       | 100000          |
| `Online social network` | 279630          |

## Pseudocode

```python
G = generate list of graphs

for all graphs:
	realV = calculate values of each node
	for all pivot strategies:
		idx = calculate pivot indexes of strategy
		for all pivot numbers:
			aprxV = calculate values of each node based on pivots
