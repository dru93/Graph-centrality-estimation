import sys
import time
import random
import numpy as np
import pandas as pd
from graph_tool.all import *

# random pivot selection
def randomPivots(G, numberOfPivots):

    nodes = list(G.get_vertices())
    pivots = random.sample(nodes, numberOfPivots)

    return pivots

# pivot selection proportional to node degree
def ranDegPivots(G, numberOfPivots):

	degreesVal = G.get_out_degrees(G.get_vertices())
	a = degreesVal

	norm = [float(i)/sum(a) for i in a]
	pivotsDegs = np.random.choice(a, numberOfPivots, replace=False, p=norm)

	dVals = list(degreesVal)
	pivotsDegs = list(pivotsDegs)
	zeroz = [0] * len(dVals)
	pivots = []
	for i in range(len(pivotsDegs)):
		for j in range(len(dVals)):
				if pivotsDegs[i] == dVals[j] and zeroz[j] == 0:
					zeroz[j] = 1
					pivots.append(j) 
	
	return pivots

# pivot selection proportional to node degree MAX
def degreePivots(G, numberOfPivots):

    degreesVal = list(G.get_out_degrees(G.get_vertices()))

    a = np.array(degreesVal)
    # sorted degrees value indexes from max to min
    sortedMaxToMinIndexes = np.argsort(a)[::-1]
    # convert to list
    pivots = sortedMaxToMinIndexes.tolist()
    # select subsection
    pivots = pivots[0:numberOfPivots]

    return pivots

# pivot selection proportional to node page rank value
def pgRankPivots(G, numberOfPivots):

    pageRankVal = list(pagerank(G))

    a = np.array(pageRankVal)
    # sorted degrees value indexes from max to min
    sortedMaxToMinIndexes = np.argsort(a)[::-1]
    # convert to list
    pivots = sortedMaxToMinIndexes.tolist()
    # select subsection
    pivots = pivots[0:numberOfPivots]

    return pivots

# pivot selection proportional to node page rank value
def pgRankReversePivots(G, numberOfPivots):

    pageRankVal = list(pagerank(G))

    a = np.array(pageRankVal)
    # sorted degrees value indexes from min to max
    sortedMaxToMinIndexes = np.argsort(a)[::]
    # convert to list
    pivots = sortedMaxToMinIndexes.tolist()
    # select subsection
    pivots = pivots[0:numberOfPivots]

    return pivots

# pivot selection by maximizing distance from previous pivot 
def maxMinPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    pivots = []
    if mixed == False:
        nodes = list(G.get_vertices())
        firstPivot = random.sample(nodes,1)[0]
        pivots.append(firstPivot)
        nodes.remove(firstPivot)
        prevPivot = firstPivot
    else:
        nodes = nodeList
        prevPivot = prevPivot

    for _ in range (numberOfPivots - 1):
        pivot = prevPivot
        maxminDistance = 0
        for node in nodes:
            distance = shortest_distance(G, source = pivot, target = node)
            if  distance > maxminDistance:
                maxminDistance = distance
                nextPivot = node
        pivots.append(nextPivot)
        nodes.remove(nextPivot)
        prevPivot = nextPivot

    return pivots

# pivot selection by maximizing sum(distances) from previous pivot 
def maxSumPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    pivots = []
    if mixed == False:
        nodes = list(G.get_vertices())
        firstPivot = random.sample(nodes,1)[0]
        pivots.append(firstPivot)
        nodes.remove(firstPivot)
        prevPivot = firstPivot
    else:
        nodes = nodeList
        prevPivot = prevPivot

    for _ in range (numberOfPivots - 1):
        pivot = prevPivot
        maxSumOfDistances = 0
        for node in nodes:
            
            sumOfDistances = len(list(all_paths(G, source = pivot, target = node, cutoff = cutoffPoint)))    
            # compare with previous max
            if  sumOfDistances > maxSumOfDistances:
                maxSumOfDistances = sumOfDistances
                nextPivot = node
        
        pivots.append(nextPivot)
        nodes.remove(nextPivot)
        prevPivot = nextPivot

    return pivots

# pivot selection by minimizing sum(distances) from previous pivot 
def minSumPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    pivots = []
    if mixed == False:
        nodes = list(G.get_vertices())
        firstPivot = random.sample(nodes,1)[0]
        pivots.append(firstPivot)
        nodes.remove(firstPivot)
        prevPivot = firstPivot
    else:
        nodes = nodeList
        prevPivot = prevPivot

    for _ in range (numberOfPivots - 1):
        pivot = prevPivot
        minSumOfDistances = sys.maxsize
        for node in nodes:
            sumOfDistances = len(list(all_paths(G, source = pivot, target = node, cutoff = 3)))
            if sumOfDistances < minSumOfDistances:
                minSumOfDistances = sumOfDistances
                nextPivot = node
        
        pivots.append(nextPivot)
        nodes.remove(nextPivot)
        prevPivot = nextPivot

    return pivots

# pivot selection by alternating maxmin, maxsum and minsum
def mixed3Pivots(G, numberOfPivots):

    pivots = []
    nodes = list(G.get_vertices())
    firstPivot = random.sample(nodes,1)[0]
    pivots.append(firstPivot)
    prevPivot = firstPivot
    nodes.remove(firstPivot)

    for i in range (numberOfPivots - 1):
        if i % 3 == 0:
            nextPivot = maxMinPivots(G, 2, True, nodes, prevPivot)
        elif i % 3 == 1:
            nextPivot = maxSumPivots(G, 2, True, nodes, prevPivot)
        elif i % 3 == 2:
            nextPivot = minSumPivots(G, 2, True, nodes, prevPivot)

        nextPivot = nextPivot[0]
        pivots.append(nextPivot)
        prevPivot = nextPivot

    return pivots

# function to make all graphs we wanna test
def makeGraphs():

    # list containing all graphs tested
    G = []

    # Erdos-Renyi random graph with 1000 nodes
    g, pos = triangulation(np.random.random((3000, 2)))
    ret = random_rewire(g, 'erdos')
    G.append(g)
    # time elapsed: ~0s

    # # US power grid graph
    # g = collection.data['power']
    # G.append(g)
    # # time elapsed: ~5s

    # Price network graph
    # g = load_graph('resources/price.xml.gz')
    # G.append(g)
    # time elapsed: ~22mins

    # # Online social network graph
    # df = pd.read_csv('resources/large.tsv', sep = '\t')
    # g = Graph(directed = True)
    # g.add_edge_list(df.values)
    # G.append(g)
    # # time elapsed: ~12h


    return G

if __name__ == "__main__":

    G = makeGraphs()

    startTime = time.time()

    graphsNames = {0:'Erdos-Renyi'}#, 1:'Power-grid', 2:'Price-Network'}#,
    				# 2:'Barabasi-Albert', 3:'Online-Social-Network'}

    # lists with ALL values calculated
    pivotValues = []
    realValues = []

    # iterate through all graphs
    for g in range(len(G)):

        graph = G[g]
        numberOfNodes = graph.num_vertices()

        nodes = list(graph.get_vertices())

        closenessOfAllNodes = list(closeness(graph))
        betweennessOfAllNodes = list(betweenness(graph)[0])

        # omit nan values
        temp = np.array(closenessOfAllNodes)
        temp = temp[np.logical_not(np.isnan(temp))]

        averageClosenessExact = np.mean(temp)
        averageBetweennessExact = np.mean(np.array(betweennessOfAllNodes))
        realValues.append([averageClosenessExact] + [averageBetweennessExact] +
                         [graphsNames[g]] + ['noPivot'] + [numberOfNodes])

        strategyNames = ['random', 'randeg', 'degree', 'pgRank', 'pgRankRev']
                    #  'maxMin', 'maxSum', 'minSum', 'mixed3'] 

        strategyDict = {0: randomPivots(graph, numberOfNodes),
                        1: ranDegPivots(graph, numberOfNodes),
                        2: degreePivots(graph, numberOfNodes),
                        3: pgRankPivots(graph, numberOfNodes),
                        4: pgRankReversePivots(graph, numberOfNodes)}
                        # : maxMinPivots(graph, numberOfNodes, False, 0, 0),
                        # : maxSumPivots(graph, numberOfNodes, False, 0, 0),
                        # : minSumPivots(graph, numberOfNodes, False, 0, 0),
                        # : mixed3Pivots(graph, numberOfNodes)}


        # iterate through all pivot strategies
        for strategy in range(0, len(strategyNames)):

            # calculate all pivot values once
            allPivots = strategyDict[strategy]
                        
            closenessOfAllPivots = []
            betweennessOfAllPivots = []
            for pivotIndex in range(numberOfNodes):
                closenessOfAllPivots.append(closenessOfAllNodes[allPivots[pivotIndex]])
                betweennessOfAllPivots.append(betweennessOfAllNodes[allPivots[pivotIndex]])

            # iterate through 1:number of nodes of graph
            for numberOfPivots in range(1, numberOfNodes):

                # omit nan values
                temp = np.array(closenessOfAllPivots[0:numberOfPivots])
                temp = temp[np.logical_not(np.isnan(temp))]

                temp2 = np.array(betweennessOfAllPivots[0:numberOfPivots])
             
                averageClosenessApprox = np.mean(temp)
                averageBetweennessApprox = np.mean(temp2)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphsNames[g]] + [strategyNames[strategy]] + [numberOfPivots])

    columnNames = ['closenessValue', 'betweennessValue', 'graphType', 'pivotStrategy', 'numberOfPivots']

    pivotValues = pd.DataFrame(pivotValues, columns = columnNames)
    realValues = pd.DataFrame(realValues, columns = columnNames)

    pivotFileName = 'results/pivotValues.csv'
    realFileName = 'results/realValues.csv'
    pivotValues.to_csv(pivotFileName, sep = ',',  index = False)
    realValues.to_csv(realFileName, sep = ',',  index = False)

    endTime = time.time()
    print('time elapsed:',int(endTime - startTime),'sec')