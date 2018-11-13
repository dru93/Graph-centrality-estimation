import sys
import time
import random
import statistics
import numpy as np
import pandas as pd
import networkx as nx

# random pivot selection
def randomPivots(G, numberOfPivots):

    nodes = list(G.nodes)
    pivots = random.sample(nodes,numberOfPivots)

    return pivots

# pivot selection proportional to node degree
def ranDegPivots(G, numberOfPivots):

    degrees = list(G.degree())
    degreesVal = [x[1] for x in degrees]

    breakpoint1 = int(np.percentile(degreesVal, 25))
    breakpoint2 = int(np.percentile(degreesVal, 50))
    breakpoint3 = int(np.percentile(degreesVal, 75))

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

# pivot selection by maximizing distance from previous pivot 
def maxMinPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    pivots = []
    if mixed == False:
        nodes = list(G.nodes)
        firstPivot = random.sample(nodes,1)[0]
        pivots.append(firstPivot)
        nodes.remove(firstPivot)
        prevPivot = firstPivot
    else:
        nodes = nodeList
        prevPivot = prevPivot

    for _ in range (0, numberOfPivots - 1):
        pivot = prevPivot
        maxminDistance = 0
        for node in nodes:
            distance = nx.shortest_path_length(G, source = pivot, target = node)
            if  distance > maxminDistance:
                maxminDistance = distance
                nextPivot = node
        pivots.append(nextPivot)
        nodes.remove(nextPivot)
        prevPivot = nextPivot

    return pivots

# pivot selection by maximizing ||distance|| from previous pivot 
def maxSumPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    pivots = []
    if mixed == False:
        nodes = list(G.nodes)
        firstPivot = random.sample(nodes,1)[0]
        pivots.append(firstPivot)
        nodes.remove(firstPivot)
        prevPivot = firstPivot
    else:
        nodes = nodeList
        prevPivot = prevPivot

    for _ in range (0, numberOfPivots - 1):
        pivot = prevPivot
        maxsumDistance = 0
        maxDistance = 0
        for node in nodes:
            distance = nx.shortest_path_length(G, source = pivot, target = node)
            if  distance > maxDistance:
                maxDistance = distance
                numberOfShortestPaths = len([p for p in nx.all_shortest_paths(G, source = pivot, target = node)])
                if numberOfShortestPaths*distance > maxsumDistance:
                    maxsumDistance = numberOfShortestPaths*distance
                    nextPivot = node
        
        pivots.append(nextPivot)
        nodes.remove(nextPivot)
        prevPivot = nextPivot

    return pivots

# pivot selection by minimizing ||distance|| from previous pivot 
def minSumPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    pivots = []
    if mixed == False:
        nodes = list(G.nodes)
        firstPivot = random.sample(nodes,1)[0]
        pivots.append(firstPivot)
        nodes.remove(firstPivot)
        prevPivot = firstPivot
    else:
        nodes = nodeList
        prevPivot = prevPivot

    for _ in range (0, numberOfPivots - 1):
        pivot = prevPivot
        minsumDistance = sys.maxsize
        maxDistance = 0
        for node in nodes:
            distance = nx.shortest_path_length(G, source = pivot, target = node)
            if  distance > maxDistance:
                maxDistance = distance
                numberOfShortestPaths = len([p for p in nx.all_shortest_paths(G, source = pivot, target = node)])
                if numberOfShortestPaths*distance < minsumDistance:
                    minsumDistance = numberOfShortestPaths*distance
                    nextPivot = node
        
        pivots.append(nextPivot)
        nodes.remove(nextPivot)
        prevPivot = nextPivot

    return pivots

# pivot selection by alternating maxmin, maxsum and minsum
def mixed3Pivots(G, numberOfPivots):

    nodes = list(G.nodes)
    pivots = []
    firstPivot = random.sample(nodes,1)[0]
    pivots.append(firstPivot)
    prevPivot = firstPivot
    nodes.remove(firstPivot)

    for i in range (0, numberOfPivots - 1):
        if i % 3 == 0:
            nextPivot = maxMinPivots(G, 2, True, nodes, prevPivot)
        elif i % 3 == 1:
            nextPivot = maxSumPivots(G, 2, True, nodes, prevPivot)
        elif i % 3 == 2:
            nextPivot = minSumPivots(G, 2, True, nodes, prevPivot)

        nextPivot = nextPivot[0]
        pivots.append(nextPivot)
        # nodes.remove(nextPivot)
        prevPivot = nextPivot

    return pivots

# pivot selection proportional to node page rank value
def pgRankPivots(G, numberOfPivots):

    pageRank = nx.pagerank(G)
    pageRank = pageRank.items() # dict to list of tuples
    pageRankVal = [x[1] for x in pageRank]

    breakpoint1 = int(np.percentile(pageRankVal, 25))
    breakpoint2 = int(np.percentile(pageRankVal, 50))
    breakpoint3 = int(np.percentile(pageRankVal, 75))

    group1 = [i for i in pageRank if i[1] <= breakpoint1]
    group2 = [i for i in pageRank if i[1] > breakpoint1 and i[1] <= breakpoint2]
    group3 = [i for i in pageRank if i[1] > breakpoint2 and i[1] <= breakpoint3]
    group4 = [i for i in pageRank if i[1] > breakpoint3]

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

# closeness centrality of nodes in graph G
def closeness(nodes, G):

    closenessNodes = []
    for node in range(0 ,len(nodes)-1):
        closenessNodes.append(nx.closeness_centrality(G, node))  
    
    return closenessNodes

# average closeness centrality of all nodes in graph G
def averageCloseness(nodes, G):

    start = time.time()
    closenessNodes = closeness(nodes, G)
    averageCloseness = statistics.mean(closenessNodes)
    end = time.time()
    elapsedTime = end - start

    print ('Value:', round(averageCloseness, 6))
    print('Time: ', round(elapsedTime, 2), 'sec')
    return averageCloseness, elapsedTime

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

# betweenness functions
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

# betweenness functions
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

# average betweenness centrality of all nodes in graph G
def averageBetweenness(nodes, G):

    start = time.time()
    betweennessNodes = betweenness(nodes, G)
    averageBetweenness = statistics.mean(betweennessNodes)
    end = time.time()
    elapsedTime = end - start

    print ('Value:', round(averageBetweenness, 6))
    print('Time: ', round(elapsedTime, 2), 'sec')

    return averageBetweenness, elapsedTime

# driver function to pff dunno I guess to drive stuff 
def driver(Graph, pivotSelection, numberOfPivots):

    if pivotSelection == 'randomPivots':
        pivots = randomPivots(Graph, numberOfPivots)
    elif pivotSelection == 'ranDegPivots':
        pivots = ranDegPivots(Graph, numberOfPivots)
    elif pivotSelection == 'maxMinPivots':
        pivots = maxMinPivots(Graph, numberOfPivots, False, 0, 0)
    elif pivotSelection == 'maxSumPivots':
        pivots = maxSumPivots(Graph, numberOfPivots, False, 0, 0)
    elif pivotSelection == 'minSumPivots':
        pivots = minSumPivots(Graph, numberOfPivots, False, 0, 0)
    elif pivotSelection == 'mixed3Pivots':
        pivots = mixed3Pivots(Graph, numberOfPivots)
    elif pivotSelection == 'pgRankPivots':
        pivots = pgRankPivots(Graph, numberOfPivots)
    elif pivotSelection == 'none':
        pivots = list(Graph.nodes)

    print('\nPivot selection strategy:', pivotSelection)
    print('Number of pivots:', numberOfPivots)

    print('\nCloseness centrality')
    closeness = averageCloseness(pivots, Graph)

    print('\nBetweenness centrality')
    betweenness = averageBetweenness(pivots, Graph)

    results = pd.DataFrame([list(closeness + betweenness)])
    columnNames = ['closeness value', 'closeness time', 'betweenness value', 'betweenness time']
    results.columns = columnNames

    return results

numberOfNodes = 1000

# Erdős-Rényi random graph
# probability for edge creation = 0.5
G = nx.fast_gnp_random_graph(numberOfNodes, 0.5, seed = 0, directed = False)

# Watts-Strogatz small-world graph 
# probability for edge rewiring = 0.5
# each node is joined with its k = 10 nearest neighbors in a ring topology
G2 = nx.watts_strogatz_graph(numberOfNodes, 10, 0.5, seed = 0)

# Barabási-Albert preferential attachment model random graph
# attaching new nodes each with m = 1000 edges that are preferentially
# attached to existing nodes with high degree
G3 = nx.barabasi_albert_graph(numberOfNodes, int(numberOfNodes/2) , seed = 0)

noPivots = driver(G, 'none', numberOfNodes)

randomPivots100 = driver(G, 'randomPivots', 100)
randomPivots50  = driver(G, 'randomPivots', 50)
randomPivots10  = driver(G, 'randomPivots', 10)

ranDegPivots100 = driver(G, 'ranDegPivots', 100)
ranDegPivots50  = driver(G, 'ranDegPivots', 50)
ranDegPivots10  = driver(G, 'ranDegPivots', 10)

maxMinPivots100 = driver(G, 'maxMinPivots', 100)
maxMinPivots50  = driver(G, 'maxMinPivots', 50)
maxMinPivots10  = driver(G, 'maxMinPivots', 10)

maxSumPivots100 = driver(G, 'maxSumPivots', 100)
maxSumPivots50  = driver(G, 'maxSumPivots', 50)
maxSumPivots10  = driver(G, 'maxSumPivots', 10)

minSumPivots100 = driver(G, 'minSumPivots', 100)
minSumPivots50  = driver(G, 'minSumPivots', 50)
minSumPivots10  = driver(G, 'minSumPivots', 10)

mixed3Pivots100 = driver(G, 'mixed3Pivots', 100)
mixed3Pivots50  = driver(G, 'mixed3Pivots', 50)
mixed3Pivots10  = driver(G, 'mixed3Pivots', 10)

pgRankPivots100 = driver(G, 'pgRankPivots', 100)
pgRankPivots50  = driver(G, 'pgRankPivots', 50)
pgRankPivots10  = driver(G, 'pgRankPivots', 10)
