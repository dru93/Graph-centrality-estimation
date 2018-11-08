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
start = time.time()

for pivot in pivots:
        pathsNumber = 0
        pathsNumberWithPivot = 0
        nodeBetweenness = []
        for n in G:
                path = nx.shortest_path(G, source = pivot, target = n)
                for pathIndex in range (0, len(path)-1):
                        if path[pathIndex] == pivot:
                                pathsNumberWithPivot += 1
                pathsNumber += 1
        nodeBetweenness.append(pathsNumberWithPivot/pathsNumber)

averageBetweennessApprox = statistics.mean(nodeBetweenness)
end = time.time()
elapsedBetweennessApprox = end - start