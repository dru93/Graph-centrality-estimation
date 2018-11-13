# Graph centrality estimation
Estimation of *closeness* and *betweenness* centrality measures based on pivot selection strategies from:

### Centrality Estimation in Large Network
 http://algo.uni-konstanz.de/publications/bp-celn-06.pdf

### Pivot selection strategies

| Strategy      | Rule                                                     |
|:-------------:|:--------------------------------------------------------:|
| `Random`      | uniformly at random                                      |
| `RanDeg`      | random proportional to degree                            |
| `MaxMin`      | non-pivot maximizing minimum distance to previous pivots |
| `MaxSum`      | non-pivot maximizing sum of distances to previous pivots |
| `MinSum`      | non-pivot minimizing sum of distances to previous pivots |
| `Mixed3`      | alternatingly MaxMin, MaxSum, and Random                 |

### Added pivot selection strategies

| Strategy      | Rule                                                     |
|:-------------:|:--------------------------------------------------------:|
| `pgRank`      | random proportional to page rank                         |

### Graphs tested

1. Random Erdős-Rényi graph with 1000 nodes
2. *TO DO:* Small worlds graph with 1000 nodes
3. *TO DO:* preferential attachment
4. *TO DO:* protein interaction
5. *TO DO:* twitter graph?
6. *TO DO:* 1st assignment graph

### Number of Pivots tested

1. Divided by 10 from original network number of nodes
2. Divided by 50 from original network number of nodes
3. Divided by 100 from original network number of nodes 