import time
import random
import statistics
import networkx as nx

# create a random Erdős-Rényi graph with 1000 nodes and probability for edge creation 0.5
G = nx.fast_gnp_random_graph(1000, 0.5, seed = 1, directed = False)

# random pivot selection
def randomPivots(G, numberOfPivots):
    nodes = list(G.nodes.keys())
    pivots = []
    for _ in range(numberOfPivots):
        pivotIndexToInput = random.randint(0, len(nodes) - 1)
        pivots.append(nodes[pivotIndexToInput])
        nodes.remove(nodes[pivotIndexToInput])
    return pivots

numberOfPivots = 100
pivots = randomPivots(G, numberOfPivots)

# exact closeness centrality
start = time.time()
closenessActualNodeValues = nx.closeness_centrality(G)
averageCloseness = statistics.mean(closenessActualNodeValues.values())
end = time.time()
elapsedCloseness = end - start

# average graph closeness approximation
start = time.time()
nodeCloseness = []

for n in G:
    distances = []
    for pivot in pivots:
        path = nx.shortest_path(G, source = n, target = pivot)
        pathLength = len(path)
        distances.append(pathLength)
    temp = numberOfPivots/sum(distances)
    nodeCloseness.append(temp)

averageClosenessApprox = statistics.mean(nodeCloseness)
end = time.time()
elapsedClosenessApprox = end - start

# betweenness functions

def single_source_shortest_path_basic(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]
    while Q:   # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:   # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma

def accumulate_endpoints(betweenness, S, P, sigma, s):
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return betweenness

def rescale(betweenness, n, normalized, directed=False, endpoints=False):
    if normalized:
        if endpoints:
            if n < 2:
                scale = None  # no normalization
            else:
                # Scale factor should include endpoint nodes
                scale = 1 / (n * (n - 1))
        elif n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

# exact betweenness centrality
start = time.time()
betweennessActualNodeValues = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G

for s in G:
    S, P, sigma = single_source_shortest_path_basic(G, s)
    betweennessActualNodeValues = accumulate_endpoints(betweennessActualNodeValues, S, P, sigma, s)
    betweennessActualNodeValues = rescale(betweennessActualNodeValues, len(G), normalized=True,
                        directed=False, endpoints=False)

averageBetweenness = statistics.mean(betweennessActualNodeValues.values())
end = time.time()
elapsedBetweenness = end - start

# approximated betweenness centrality
start = time.time()
betweennessApproxNodeValues = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G

for s in pivots:
    S, P, sigma = single_source_shortest_path_basic(G, s)
    betweennessApproxNodeValues = accumulate_endpoints(betweennessApproxNodeValues, S, P, sigma, s)
    betweennessApproxNodeValues = rescale(betweennessApproxNodeValues, len(pivots), normalized=True,
                        directed=False, endpoints=False)

averageBetweennessApprox = statistics.mean(betweennessApproxNodeValues.values())
end = time.time()
elapsedBetweennessApprox = end - start

print('\n\t~~~ Time tracks ~~~')
print('Betweenness centrality:\t\t\t', round(elapsedBetweenness,2), 's')
print('Betweenness centrality approximation:\t', round(elapsedBetweennessApprox,2), 's')
print('Closeness centrality:\t\t\t', round(elapsedCloseness,2), 's')
print('Closeness centrality approximation:\t', round(elapsedClosenessApprox,2), 's')

print('\n\t~~~ Centrality values ~~~')
print('Betweenness centrality:\t\t\t', round(averageBetweenness,6))
print('Betweenness centrality approximation:\t', round(averageBetweennessApprox,6))
print('Closeness centrality:\t\t\t', round(averageCloseness,6))
print('Closeness centrality approximation:\t', round(averageClosenessApprox,6))