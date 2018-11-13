import time
import random
import statistics
import numpy as np
import networkx as nx

numberOfNodes = 100

# create a random Erdős-Rényi graph with 1000 nodes and probability for edge creation 0.5
G = nx.fast_gnp_random_graph(numberOfNodes, 0.5, seed = 1, directed = False)

# random pivot selection
numberOfPivots = int(len(G)/10)

def randomPivots(G, numberOfPivots):
    nodes = list(G.nodes)
    pivots = random.sample(nodes,numberOfPivots)
    return pivots

def degreePivots(G, numberOfPivots):
    degrees = list(G.degree())
    degreesVal = [x[1] for x in degrees]

    breakpoint1 = int(np.percentile(degreesVal,25))
    breakpoint2 = int(np.percentile(degreesVal,50))
    breakpoint3 = int(np.percentile(degreesVal,75))

    group1 = [i for i in degrees if i[1] <= breakpoint1]
    group2 = [i for i in degrees if i[1] > breakpoint1 and i[1] <= breakpoint2]
    group3 = [i for i in degrees if i[1] > breakpoint2 and i[1] <= breakpoint3]
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

def MaxminPivots(G, numberOfPivots):
    nodes = list(G.nodes)
    pivots = []
    firstPivot = random.sample(nodes,1)[0]
    pivots.append(firstPivot)
    nodes.remove(firstPivot)
    prevPivot = firstPivot
    for _ in range (0, numberOfPivots - 1):
        pivot = prevPivot
        maxDistance = 0
        for node in nodes:
            distance = nx.shortest_path_length(G, source = pivot, target = node)
            if  distance > maxDistance:
                maxDistance = distance
                nextPivot = node
        pivots.append(nextPivot)
        nodes.remove(nextPivot)
        prevPivot = nextPivot
    return pivots

pivots = MaxminPivots(G, numberOfPivots)

# closeness centrality of nodes in graph G
def closeness(nodes, G):
    closenessNodes = []
    for node in range(0 ,len(nodes)-1):
        closenessNodes.append(nx.closeness_centrality(G, node))  
    
    return closenessNodes

# exact closeness centrality
start = time.time()
closenessAllNodes = closeness(list(G.nodes), G)
averageCloseness = statistics.mean(closenessAllNodes)
end = time.time()
elapsedCloseness = end - start

# average graph closeness approximation
start = time.time()
closenessPivots = closeness(pivots, G)
averageClosenessApprox = statistics.mean(closenessPivots)
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

# betweenness centrality of nodes in graph G
def betweenness(nodes, G):
    betweennessNodes = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    
    for s in nodes:
        S, P, sigma = single_source_shortest_path_basic(G, s)
        betweennessNodes = accumulate_endpoints(betweennessNodes, S, P, sigma, s)
        betweennessNodes = rescale(betweennessNodes, len(G), normalized=True,
                            directed=False, endpoints=False)
    return betweennessNodes.values()

# exact betweenness centrality
start = time.time()
betweennessAllNodeValues = betweenness(list(G.nodes), G)
averageBetweenness = statistics.mean(betweennessAllNodeValues)
end = time.time()
elapsedBetweenness = end - start

# approximated betweenness centrality
start = time.time()
betweennessPivotValues = betweenness(pivots, G)
averageBetweennessApprox = statistics.mean(betweennessPivotValues)
end = time.time()
elapsedBetweennessApprox = end - start

print('\nNumber of pivots:', numberOfPivots)

print('\n\t~~~ Time tracks (sec) ~~~')
print('Betweenness centrality:\t\t\t', round(elapsedBetweenness,2))
print('Betweenness centrality approximation:\t', round(elapsedBetweennessApprox,2))
print('Closeness centrality:\t\t\t', round(elapsedCloseness,2))
print('Closeness centrality approximation:\t', round(elapsedClosenessApprox,2))

print('\n\t~~~ Centrality values ~~~')
print('Betweenness centrality:\t\t\t', round(averageBetweenness,6))
print('Betweenness centrality approximation:\t', round(averageBetweennessApprox,6))
print('Closeness centrality:\t\t\t', round(averageCloseness,6))
print('Closeness centrality approximation:\t', round(averageClosenessApprox,6))

# TO DO
# maxmin pivots
# maxsum pivots
# minsum pivots
# mixed pivots
# random proportional to pagerank pivots