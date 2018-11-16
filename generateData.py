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
            # find optimal cutoff point
            cutoffPoint = G.num_vertices()
            notEnoughPivots = True
            while notEnoughPivots:
                sumOfDistances = len(list(all_paths(G, source = pivot, target = node, cutoff = cutoffPoint)))
                if sumOfDistances > numberOfPivots:
                    notEnoughPivots = False
            
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
def makeGraphs(numberOfNodes):

    # list containing all graphs tested
    G = []

    # Erdos-Renyi random graph
    g, pos = triangulation(np.random.random((numberOfNodes, 2)))
    ret = random_rewire(g, 'erdos')
    G.append(g)

    # US power grid graph
    g = collection.data['power']
    G.append(g)

    # Price network graph
    g = load_graph('price.xml.gz')
    G.append(g)
    # Elapsed time: 1341 sec

    return G

if __name__ == "__main__":

    numberOfRandomGraphNodes = 1000
    G = makeGraphs(numberOfRandomGraphNodes)

    startTime = time.time()

    graphsNames = {0:'Erdos-Renyi', 1:'Power-grid', 2:'Price-Network'}#,
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

        strategyNames = ['random', 'ranDeg', 'pgRank']#, 
                    #  'maxMin', 'maxSum', 'minSum', 'mixed3'] 

        # iterate through all pivot strategies
        for strategy in range(0, len(strategyNames)):

            # calculate all pivot values once
            allPivots = []

            if strategy == 0:
                allPivots = randomPivots(graph, numberOfNodes)
            elif strategy == 1:
                allPivots = ranDegPivots(G[g], numberOfNodes)
            elif strategy == 2:
                allPivots = pgRankPivots(G[g], numberOfNodes)
            # elif strategy == 3:
            #     allPivots = maxMinPivots(G[g], numberOfNodes, False, 0, 0)
            # elif strategy == 4:
            #     allPivots = maxSumPivots(G[g], numberOfNodes, False, 0, 0)
            # elif strategy == 5:
            #     allPivots = minSumPivots(G[g], numberOfNodes, False, 0, 0)
            # elif strategy == 6:
            #     allPivots = mixed3Pivots(G[g], numberOfNodes)
            
            closenessOfAllPivots = []
            betweennessOfAllPivots = []
            for pivotIndex in range(numberOfNodes):
                closenessOfAllPivots.append(closenessOfAllNodes[allPivots[pivotIndex]])
                betweennessOfAllPivots.append(betweennessOfAllNodes[allPivots[pivotIndex]])

            # iterate through 1:number of nodes of graph
            for numberOfPivots in range(1, numberOfNodes):
                closenessSumOfAllPivots = sum(closenessOfAllPivots[0:numberOfPivots])
                betweennessSumOfAllPivots = sum(betweennessOfAllPivots[0:numberOfPivots])

                averageClosenessApprox = closenessSumOfAllPivots/numberOfPivots
                averageBetweennessApprox = betweennessSumOfAllPivots/numberOfPivots

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphsNames[g]] + [strategyNames[strategy]] + [numberOfPivots])

    columnNames = ['closenessValue', 'betweennessValue', 'graphType', 'pivotStrategy', 'numberOfPivots']

    pivotValues = pd.DataFrame(pivotValues, columns=columnNames)
    realValues = pd.DataFrame(realValues, columns=columnNames)

    pivotFileName = 'pivotValues.csv'
    realFileName = 'realValues.csv'
    pivotValues.to_csv(pivotFileName, sep=',',  index = False)
    realValues.to_csv(realFileName, sep=',',  index = False)

    endTime = time.time()
    print('Elapsed time:',int(endTime - startTime),'sec')


# to do 
# def stratifiedPivots(G, numberOfPivots):