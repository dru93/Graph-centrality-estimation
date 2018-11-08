import time
import random
import statistics
import networkx as nx

# create a random Erdős-Rényi graph with 1000 nodes and probability for edge creation 0.5
G = nx.fast_gnp_random_graph(1000, 0.5, seed = 1, directed = False)

# closeness centrality
start = time.time()
closenessActualNodeValues = nx.closeness_centrality(G)
averageCloseness = statistics.mean(closenessActualNodeValues.values())
end = time.time()
elapsedCloseness = end - start

# betweenness centrality
start = time.time()
betweennessActualNodeValues = nx.betweenness_centrality(G)
averageBetweenness = statistics.mean(betweennessActualNodeValues.values())
end = time.time()
elapsedBetweenness = end - start

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

# average graph betweenness approximation
# calculate distance from pivot to all nodes t
# betweenness of u = to sum(sppps from pivot to t that contain u)/sum(different shortest pivot-u paths)


def _single_source_shortest_path_basic(G, s):
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

def _accumulate_endpoints(betweenness, S, P, sigma, s):
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

def _rescale(betweenness, n, normalized,
             directed=False, k=None, endpoints=False):
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
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

start = time.time()

betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
nodes = pivots
for s in nodes:
        S, P, sigma = _single_source_shortest_path_basic(G, s)
        betweenness = _accumulate_endpoints(betweenness, S, P, sigma, s)
        betweenness = _rescale(betweenness, len(G), normalized=True,
                           directed=False, k=None, endpoints=False)

averageBetweennessApprox = statistics.mean(betweenness.values())
end = time.time()
elapsedBetweennessApprox = end - start

print(elapsedBetweenness, elapsedCloseness, elapsedBetweennessApprox, elapsedClosenessApprox)
print(averageBetweenness, averageBetweennessApprox, averageCloseness, averageClosenessApprox)