import sys
import time
import random
import statistics
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from heapq import nlargest

# random pivot selection
def randomPivots(G, numberOfPivots):

    nodes = list(G.nodes)
    pivots = random.sample(nodes, numberOfPivots)

    return pivots

# pivot selection proportional to node degree
def ranDegPivots(G, numberOfPivots):

    degrees = list(G.degree())
    degreesVal = [x[1] for x in degrees]
    
    a = np.array(degreesVal)
    # sorted degrees value indexes from max to min
    sortedMaxToMinIndexes = np.argsort(a)[::-1]
    # convert to list
    pivots = sortedMaxToMinIndexes.tolist()
    # select subsection
    pivots = pivots[0:numberOfPivots]

    return pivots

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

    a = np.array(pageRankVal)
    # sorted degrees value indexes from max to min
    sortedMaxToMinIndexes = np.argsort(a)[::-1]
    # convert to list
    pivots = sortedMaxToMinIndexes.tolist()
    # select subsection
    pivots = pivots[0:numberOfPivots]

    return pivots

# closeness centrality of nodes in graph G
def closeness(nodes, G):

    closenessNodes = []
    for node in range(len(nodes)):
        closenessNodes.append(nx.closeness_centrality(G, nodes[node]))
    
    return closenessNodes

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
def accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness

# betweenness functions
def rescale(betweenness, n, normalized, directed = False, k = None, endpoints = False):

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

# betweenness centrality of nodes in graph G
def betweenness(nodes, G):

    betweennessNodes = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
        
    for s in nodes:
        S, P, sigma = single_source_shortest_path_basic(G, s)
        betweennessNodes = accumulate_basic(betweennessNodes, S, P, sigma, s)
        betweennessNodes = rescale(betweennessNodes, len(G), normalized = True,
                            directed = False, k = len(nodes), endpoints = False)
    
    return list(betweennessNodes.values())

# function to make all graphs we wanna test
def makeGraphs(numberOfNodes):

	# list containing all graphs tested
	G = []

	# G[0]: Erdos-Renyi random graph
	# probability for edge creation = 0.5
	G.append(nx.fast_gnp_random_graph(numberOfNodes, 0.5, seed = 0, directed = False))

	# G[1]: Watts-Strogatz small-world graph 
	# probability for edge rewiring = 0.5
	# each node is joined with its k = 10 nearest neighbors in a ring topology
	G.append(nx.watts_strogatz_graph(numberOfNodes, 10, 0.5, seed = 0))

	# G[2]: Barabasi-Albert preferential attachment model random graph
	# Number of edges to attach from a new node to existing nodes = numberOfNodes/2
	G.append(nx.barabasi_albert_graph(numberOfNodes, int(numberOfNodes/2) , seed = 0))

	# # G[3]: Online Social Network example (assignment 1) undirected
	# # medium.tsv has 2862 nodes, large.tsv has 279630 nodes
	# GnotConnected = nx.read_edgelist('data/s22078031/large.tsv', create_using = nx.Graph())
	# # original graph is not connected
	# # centrality measures are calculated only on connected graphs
	# # so we will append the largest connected component of G[3]
	# Gconnected = max(nx.connected_component_subgraphs(GnotConnected), key=len)
	# G.append(Gconnected)

	return G

if __name__ == "__main__":

    startTime = time.time()

    numberOfRandomGraphNodes = 100
    G = makeGraphs(numberOfRandomGraphNodes)
    # 25, runtime: 2sec
    # 100, runtime: 174 sec

    graphsList = {0:'Erdos-Renyi', 1:'Watts-Strogatz',
    				2:'Barabasi-Albert', 3:'Online-Social-Network'}

    # lists with ALL values calculated
    pivotValues = []
    realValues = []

    # # iterate through all graphs
    for g in range(len(G)):

            # graphType will hold the name of the current graph
            graphName = graphsList[g]
            nodes = list(G[g].nodes)

            closenessOfAllNodes = closeness(nodes, G[g])
            betweennessOfAllNodes = betweenness(nodes, G[g])

            averageClosenessExact = statistics.mean(closenessOfAllNodes)
            averageBetweennessExact = statistics.mean(betweennessOfAllNodes)
            realValues.append([averageClosenessExact] + [averageBetweennessExact] + [graphName] + ['noPivot'] + [len(G[g])])

            # do for 1 node to all nodes
            for numberOfPivots in range(1, len(G[g])+1):

                pivots = randomPivots(G[g], numberOfPivots)

                closenessOfAllPivots = []
                betweennessOfAllPivots = []
                for pivot in pivots:
                    closenessOfAllPivots.append(closenessOfAllNodes[pivot])
                    betweennessOfAllPivots.append(betweennessOfAllNodes[pivot])

                averageClosenessApprox = statistics.mean(closenessOfAllPivots)
                averageBetweennessApprox = statistics.mean(betweennessOfAllPivots)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphName] + ['randomPivot'] + [numberOfPivots])


                pivots = ranDegPivots(G[g], numberOfPivots)

                closenessOfAllPivots = []
                betweennessOfAllPivots = []
                for pivot in pivots:
                    closenessOfAllPivots.append(closenessOfAllNodes[pivot])
                    betweennessOfAllPivots.append(betweennessOfAllNodes[pivot])

                averageClosenessApprox = statistics.mean(closenessOfAllPivots)
                averageBetweennessApprox = statistics.mean(betweennessOfAllPivots)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphName] + ['ranDegPivot'] + [numberOfPivots])


                pivots = maxMinPivots(G[g], numberOfPivots, False, 0, 0)

                closenessOfAllPivots = []
                betweennessOfAllPivots = []
                for pivot in pivots:
                    closenessOfAllPivots.append(closenessOfAllNodes[pivot])
                    betweennessOfAllPivots.append(betweennessOfAllNodes[pivot])

                averageClosenessApprox = statistics.mean(closenessOfAllPivots)
                averageBetweennessApprox = statistics.mean(betweennessOfAllPivots)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphName] + ['maxMinPivot'] + [numberOfPivots])


                pivots = maxSumPivots(G[g], numberOfPivots, False, 0, 0)

                closenessOfAllPivots = []
                betweennessOfAllPivots = []
                for pivot in pivots:
                    closenessOfAllPivots.append(closenessOfAllNodes[pivot])
                    betweennessOfAllPivots.append(betweennessOfAllNodes[pivot])

                averageClosenessApprox = statistics.mean(closenessOfAllPivots)
                averageBetweennessApprox = statistics.mean(betweennessOfAllPivots)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphName] + ['maxSumPivot'] + [numberOfPivots])


                pivots = minSumPivots(G[g], numberOfPivots, False, 0, 0)

                closenessOfAllPivots = []
                betweennessOfAllPivots = []
                for pivot in pivots:
                    closenessOfAllPivots.append(closenessOfAllNodes[pivot])
                    betweennessOfAllPivots.append(betweennessOfAllNodes[pivot])

                averageClosenessApprox = statistics.mean(closenessOfAllPivots)
                averageBetweennessApprox = statistics.mean(betweennessOfAllPivots)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphName] + ['minSumPivots'] + [numberOfPivots])


                pivots = mixed3Pivots(G[g], numberOfPivots)

                closenessOfAllPivots = []
                betweennessOfAllPivots = []
                for pivot in pivots:
                    closenessOfAllPivots.append(closenessOfAllNodes[pivot])
                    betweennessOfAllPivots.append(betweennessOfAllNodes[pivot])

                averageClosenessApprox = statistics.mean(closenessOfAllPivots)
                averageBetweennessApprox = statistics.mean(betweennessOfAllPivots)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphName] + ['mixed3Pivots'] + [numberOfPivots])


                pivots = pgRankPivots(G[g], numberOfPivots)

                closenessOfAllPivots = []
                betweennessOfAllPivots = []
                for pivot in pivots:
                    closenessOfAllPivots.append(closenessOfAllNodes[pivot])
                    betweennessOfAllPivots.append(betweennessOfAllNodes[pivot])

                averageClosenessApprox = statistics.mean(closenessOfAllPivots)
                averageBetweennessApprox = statistics.mean(betweennessOfAllPivots)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphName] + ['pgRankPivots'] + [numberOfPivots])

    columnNames = ['closenessValue', 'betweennessValue', 'graphType', 'pivotStrategy', 'numberOfPivots']

    pivotValues = pd.DataFrame(pivotValues, columns=columnNames)
    realValues = pd.DataFrame(realValues, columns=columnNames)

    pivotFileName = 'pivotValues-' + str(numberOfRandomGraphNodes) + 'n.csv'
    realFileName = 'realValues-' + str(numberOfRandomGraphNodes) + 'n.csv'
    pivotValues.to_csv(pivotFileName, sep=',',  index = False)
    realValues.to_csv(realFileName, sep=',',  index = False)

    endTime = time.time()
    print('Elapsed time:',int(endTime - startTime),'sec')
