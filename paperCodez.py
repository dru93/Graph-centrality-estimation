import time
import random
import statistics
import numpy as np
import networkx as nx

# create a random Erdős-Rényi graph with 1000 nodes and probability for edge creation 0.5
G = nx.fast_gnp_random_graph(1000, 0.5, seed = 1, directed = False)

# random pivot selection

def randomPivots(G, numberOfPivots):
    nodes = list(G.nodes.keys())
    pivots = random.sample(nodes,numberOfPivots)
    return pivots

def degreePivots(G, numberOfPivots):
    degrees = list(G.degree())
    degreesVal = [x[1] for x in degrees]

    breakpoint1 = int(np.percentile(degreesVal,25))
    breakpoint2 = int(np.percentile(degreesVal,50))
    breakpoint3 = int(np.percentile(degreesVal,75))

    group1 = [i for i in degrees if i[1] <= breakpoint1]
    group2 = [i for i in degrees if i[1] > breakpoint1 and i <= breakpoint2]
    group3 = [i for i in degrees if i[1] > breakpoint2 and i <= breakpoint3]
    group4 = [i for i in degrees if i[1] > breakpoint3]

    nOfSamplesGroup1 = round(len(group1) * numberOfPivots/len(G), 0)
    nOfSamplesGroup2 = round(len(group2) * numberOfPivots/len(G), 0)
    nOfSamplesGroup3 = round(len(group3) * numberOfPivots/len(G), 0)
    nOfSamplesGroup4 = round(len(group4) * numberOfPivots/len(G), 0)

    pivots1 = random.sample([x[0] for x in group1], int(nOfSamplesGroup1))
    pivots2 = random.sample([x[0] for x in group2], int(nOfSamplesGroup2))
    pivots3 = random.sample([x[0] for x in group3], int(nOfSamplesGroup3))
    pivots4 = random.sample([x[0] for x in group4], int(nOfSamplesGroup4))
    allPivotDegrees = pivots1 + pivots2 + pivots3 + pivots4
    return allPivotDegrees


    
numberOfPivots = 1000
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

print('\nNumber of pivots:', numberOfPivots)

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

# counter = 0
# for i in range (0, len(degrees)-1):
#     if degrees[i] >= statistics.mean(degrees):
#         counter += 1
