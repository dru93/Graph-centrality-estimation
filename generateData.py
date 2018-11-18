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

    norm = np.array([float(i)/sum(a) for i in a])
    # make 0 smallest numb
    for i in range(len(norm)):
        norm[i] = max(sys.float_info.min, norm[i])
    pivotsDegs = np.random.choice(a, numberOfPivots, replace=False, p=norm)

    zeroz = [0] * len(degreesVal)
    pivots = []
    for i in range(len(pivotsDegs)):
        for j in range(len(degreesVal)):
                if pivotsDegs[i] == degreesVal[j] and zeroz[j] == 0:
                    zeroz[j] = 1
                    pivots.append(j) 

    return pivots

# pivot selection proportional to node page rank
def ranPgRankPivots(G, numberOfPivots):

    pageRankVal = list(pagerank(G))
    a = pageRankVal

    norm = np.array([float(i)/sum(a) for i in a])
    # make 0 smallest numb
    for i in range(len(norm)):
        norm[i] = max(sys.float_info.min, norm[i])
    pivotsPgRank = np.random.choice(a, numberOfPivots, replace=False, p=norm)

    zeroz = [0] * len(pageRankVal)
    pivots = []
    for i in range(len(pivotsPgRank)):
        for j in range(len(pageRankVal)):
                if pivotsPgRank[i] == pageRankVal[j] and zeroz[j] == 0:
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

# pivot selection by alternating pgRank and pgRankRev
def pgRankAlternatePivots(G, numberOfPivots):
    pageRankVal = list(pagerank(G))
    
    a = np.array(pageRankVal)
    # sorted degrees value indexes from min to max
    sortedMaxToMinIndexes = np.argsort(a)[::]
    # convert to list
    pivotsSorted = sortedMaxToMinIndexes.tolist()
    
    pivots = []
    # max then min then max etc alternating
    for pivot in range(numberOfPivots):
        if pivot % 2 == 0:
            pivots.append(pivotsSorted[0])
            pivotsSorted.remove(pivotsSorted[0])
        else:
            pivots.append(pivotsSorted[len(pivotsSorted)-1])
            pivotsSorted.remove(pivotsSorted[len(pivotsSorted)-1])        

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
            
            sumOfDistances = len(list(all_paths(G, source = pivot, target = node)))    
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
            sumOfDistances = len(list(all_paths(G, source = pivot, target = node)))
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

    numberOfNodes = 20#pow(10,2)
    # 380 sec: pow(10,5), G1-3
    # pow(10,6): Segmentation fault (core dumped)
    # with distances: 18, 120sec
    # with distances: 20, 360sec

    # Erdos-Renyi random graph with n nodes
    g, pos = triangulation(np.random.random((numberOfNodes, 2)))
    ret = random_rewire(g, 'erdos')
    G.append(g)
    # time elapsed: ~80s, n = 10000

    # Price network graph
    g =  price_network(numberOfNodes)
    G.append(g)
    # time elapsed: ~ok

    # Barabasi-Albert graph
    g =  price_network(numberOfNodes, directed = False)
    G.append(g)
    # time elapsed: ~22mins

    # US power grid graph
    g = collection.data['power']
    G.append(g)
    # time elapsed: ~10s

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

    graphsNames = ['Erdos-Renyi']#, 'Price-Network', 'Barabasi-Albert']
                    # 'Power-grid']
    				# 'Online-Social-Network']

    # lists with ALL values calculated
    pivotValues = []
    realValues = []
    columnNames = ['closenessValue', 'betweennessValue', 'graphType',
                    'pivotStrategy', 'numberOfPivots']

    # iterate through all graphs
    for g in range(len(graphsNames)):

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

        strategyNames = ['random', 'ranDeg', 'ranPgRank', 
                        'degree', 'pgRank', 'pgRankRev', 'pgRankAlt']
                        #  'maxMin', 'maxSum', 'minSum', 'mixed3'] 

        strategyDict = {'random': randomPivots(graph, numberOfNodes),
                        'ranDeg': ranDegPivots(graph, numberOfNodes),
                        'ranPgRank': ranPgRankPivots(graph, numberOfNodes),
                        'degree': degreePivots(graph, numberOfNodes),
                        'pgRank': pgRankPivots(graph, numberOfNodes),
                        'pgRankRev': pgRankReversePivots(graph, numberOfNodes),
                        'pgRankAlt': pgRankAlternatePivots(graph, numberOfNodes),
                        'maxMin': maxMinPivots(graph, numberOfNodes, False, 0, 0),
                        'maxSum': maxSumPivots(graph, numberOfNodes, False, 0, 0),
                        'minSum': minSumPivots(graph, numberOfNodes, False, 0, 0),
                        'mixed3': mixed3Pivots(graph, numberOfNodes)}


        # iterate through all pivot strategies
        for strategy in strategyNames:

            # calculate all pivot values once
            allPivots = strategyDict[strategy]
                        
            closenessOfAllPivots = []
            betweennessOfAllPivots = []
            for pivotIndex in range(numberOfNodes):
                closenessOfAllPivots.append(closenessOfAllNodes[allPivots[pivotIndex]])
                betweennessOfAllPivots.append(betweennessOfAllNodes[allPivots[pivotIndex]])

            # iterate through 1:number of nodes of graph
            for numberOfPivots in range(1, numberOfNodes+1):

                # omit nan values
                temp = np.array(closenessOfAllPivots[0:numberOfPivots])
                temp = temp[np.logical_not(np.isnan(temp))]

                temp2 = np.array(betweennessOfAllPivots[0:numberOfPivots])
             
                averageClosenessApprox = np.mean(temp)
                averageBetweennessApprox = np.mean(temp2)

                pivotValues.append([averageClosenessApprox] + [averageBetweennessApprox] +
                                [graphsNames[g]] + [strategy] + [numberOfPivots])


    pivotValues = pd.DataFrame(pivotValues, columns = columnNames)
    realValues = pd.DataFrame(realValues, columns = columnNames)

    pivotFileName = 'results/pivotValues.csv'
    realFileName = 'results/realValues.csv'
    pivotValues.to_csv(pivotFileName, sep = ',',  index = False)
    realValues.to_csv(realFileName, sep = ',',  index = False)



    endTime = time.time()
    print('time elapsed:',int(endTime - startTime),'sec')