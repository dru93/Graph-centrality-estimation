import sys
import time
import random
import numpy as np
import pandas as pd
from graph_tool.all import *

'''
Generate graphs
Compute closeness and betweenness values for each node
Select pivots by different pivot selection strategies
'''

# random pivot selection
def randomPivots(G, numberOfPivots, mixed):

    start = time.time()
    nodes = list(G.get_vertices())
    pivots = random.sample(nodes, numberOfPivots)
    end = time.time()
    if mixed == False:
        print('\t\tRandom pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection proportional to node degree
def ranDegPivots(G, numberOfPivots):
    
    start = time.time()

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

    end = time.time()
    print('\t\tRandom degree pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection proportional to node page rank
def ranPgRankPivots(G, numberOfPivots):

    start = time.time()

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

    end = time.time()
    print('\t\tRandom page rank pivot selection:', round(end-start, 4), 'sec')
    
    return pivots

# pivot selection proportional to node degree MAX
def degreePivots(G, numberOfPivots):

    start = time.time()

    degreesVal = list(G.get_out_degrees(G.get_vertices()))

    a = np.array(degreesVal)
    # sorted degrees value indexes from max to min
    sortedMaxToMinIndexes = np.argsort(a)[::-1]
    # convert to list
    pivots = sortedMaxToMinIndexes.tolist()
    # select subsection
    pivots = pivots[0:numberOfPivots]

    end = time.time()
    print('\t\tDegree pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection proportional to node page rank value
def pgRankPivots(G, numberOfPivots):

    start = time.time()

    pageRankVal = list(pagerank(G))

    a = np.array(pageRankVal)
    # sorted degrees value indexes from max to min
    sortedMaxToMinIndexes = np.argsort(a)[::-1]
    # convert to list
    pivots = sortedMaxToMinIndexes.tolist()
    # select subsection
    pivots = pivots[0:numberOfPivots]

    end = time.time()
    print('\t\tPage rank pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection proportional to node page rank value
def pgRankReversePivots(G, numberOfPivots):

    start = time.time()

    pageRankVal = list(pagerank(G))

    a = np.array(pageRankVal)
    # sorted degrees value indexes from min to max
    sortedMaxToMinIndexes = np.argsort(a)[::]
    # convert to list
    pivots = sortedMaxToMinIndexes.tolist()
    # select subsection
    pivots = pivots[0:numberOfPivots]

    end = time.time()
    print('\t\tPage rank reverse pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection by alternating pgRank and pgRankRev
def pgRankAlternatePivots(G, numberOfPivots):

    start = time.time()

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

    end = time.time()
    print('\t\tPage rank alternate pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection by maximizing distance from previous pivot 
def maxMinPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    start = time.time()

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

    end = time.time()
    if mixed == False:
        print('\t\tMaxmin pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection by maximizing sum(distances) from previous pivot 
def maxSumPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    start = time.time()

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

    end = time.time()

    if mixed == False:
        print('\t\tMaxsum pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection by minimizing sum(distances) from previous pivot 
def minSumPivots(G, numberOfPivots, mixed, nodeList, prevPivot):

    start = time.time()

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

    end = time.time()
    if mixed == False:
        print('\t\tMinsum pivot selection:', round(end-start, 4), 'sec')

    return pivots

# pivot selection by alternating random, maxsum and minsum
def mixed3Pivots(G, numberOfPivots):

    start = time.time()

    pivots = []
    nodes = list(G.get_vertices())
    firstPivot = random.sample(nodes,1)[0]
    pivots.append(firstPivot)
    prevPivot = firstPivot
    nodes.remove(firstPivot)

    for i in range (numberOfPivots - 1):
        if i % 3 == 0:
            nextPivot = randomPivots(G, 1, True)
        elif i % 3 == 1:
            nextPivot = maxSumPivots(G, 2, True, nodes, prevPivot)
        elif i % 3 == 2:
            nextPivot = minSumPivots(G, 2, True, nodes, prevPivot)

        nextPivot = nextPivot[0]
        pivots.append(nextPivot)
        prevPivot = nextPivot

    end = time.time()
    print('\t\tMixed3 pivot selection:', round(end-start, 4), 'sec')

    return pivots


# function to make all graphs we wanna test
def makeGraphs(numberOfNodes):

    # list containing all graphs tested
    G = []

    # Erdos-Renyi random graph with n nodes
    g, pos = triangulation(np.random.random((numberOfNodes, 2)))
    ret = random_rewire(g, 'erdos')
    G.append(g)

    # Barabasi-Albert graph
    g =  price_network(numberOfNodes, directed = False)
    G.append(g)

    # US power grid graph
    g = collection.data['power']
    G.append(g)

    # # Online social network graph
    # df = pd.read_csv('resources/large.tsv', sep = '\t')
    # g = Graph(directed = True)
    # g.add_edge_list(df.values)
    # G.append(g)

    return G

def drawGraph(g, graphName):
    
    graph_draw(g, pos = sfdp_layout(g, cooling_step=0.99),
            vertex_fill_color = g.vertex_index, vertex_size=2,
            edge_pen_width=1,
            output = 'images/' + str(g.num_vertices()) + '-' + graphName + '.png')

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        numberOfNodes = int(input('Select number of nodes for random graphs: '))
    else:
        numberOfNodes = int(sys.argv[1])

    start = time.time()
    G = makeGraphs(numberOfNodes)
    end = time.time()
    print('Making graphs:', round(end-start, 4), 'sec')

    startGlobal = time.time()

    graphsNames = ['Erdos-Renyi', 'Barabasi-Albert', 'Power-grid']

    # lists with ALL values calculated
    pivotValues = []
    realValues = []
    columnNames = ['closenessValue', 'betweennessValue', 'graphType',
                    'pivotStrategy', 'numberOfPivots']

    # iterate through all graphs
    for g in range(len(graphsNames)):

        print(graphsNames[g], 'Graph calculations')

        graph = G[g]
        numberOfNodes = graph.num_vertices()

        # # draw graphs
        # start = time.time()
        # drawGraph(graph, graphsNames[g])
        # end = time.time()
        # print('\tDrawing', round(end-start, 4), 'sec')

        start = time.time()
        closenessOfAllNodes = list(closeness(graph))
        end = time.time()
        print('\tAverage closeness centrality', round(end-start, 4), 'sec')
        start = time.time()
        betweennessOfAllNodes = list(betweenness(graph)[0])
        end = time.time()
        print('\tAverage betweenness centrality', round(end-start, 4), 'sec')

        # omit nan values
        temp = np.array(closenessOfAllNodes)
        temp = temp[np.logical_not(np.isnan(temp))]

        averageClosenessExact = np.mean(temp)
        averageBetweennessExact = np.mean(np.array(betweennessOfAllNodes))
        realValues.append([averageClosenessExact] + [averageBetweennessExact] +
                         [graphsNames[g]] + ['noPivot'] + [numberOfNodes])

        strategyNames = ['random', 'ranDeg', 'ranPgRank', 
                        'degree', 'pgRank', 'pgRankRev', 'pgRankAlt']#,
                        #'maxMin', 'maxSum', 'minSum', 'mixed3'] 

        print ('\tPivots:')

        strategyDict = {'random': randomPivots(graph, numberOfNodes, False),
                        'ranDeg': ranDegPivots(graph, numberOfNodes),
                        'ranPgRank': ranPgRankPivots(graph, numberOfNodes),
                        'degree': degreePivots(graph, numberOfNodes),
                        'pgRank': pgRankPivots(graph, numberOfNodes),
                        'pgRankRev': pgRankReversePivots(graph, numberOfNodes),
                        'pgRankAlt': pgRankAlternatePivots(graph, numberOfNodes)}
                        # 'maxMin': maxMinPivots(graph, numberOfNodes, False, 0, 0), # slow
                        # 'maxSum': maxSumPivots(graph, numberOfNodes, False, 0, 0), # slow
                        # 'minSum': minSumPivots(graph, numberOfNodes, False, 0, 0), # slow
                        # 'mixed3': mixed3Pivots(graph, numberOfNodes), # slow

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

    pivotFileName = 'tables/pivotValues.csv'
    realFileName = 'tables/realValues.csv'
    pivotValues.to_csv(pivotFileName, sep = ',',  index = False)
    realValues.to_csv(realFileName, sep = ',',  index = False)

    endGlobal = time.time()
    print('Total time elapsed:',int(endGlobal - startGlobal),'sec')

'''
RUNTIMES

~5.000 nodes
Power-grid Graph calculations
        Average closeness centrality 3.4199 sec
        Average betweenness centrality 3.4962 sec
        Pivots:
                Random pivot selection: 0.0098 sec
                Random degree pivot selection: 40.682 sec
                Random page rank pivot selection: 23.0815 sec
                Degree pivot selection: 0.0027 sec
                Page rank pivot selection: 4.2352 sec
                Page rank reverse pivot selection: 4.2566 sec
                Page rank alternate pivot selection: 4.5548 sec

10.000 nodes
Erdos-Renyi Graph calculations
        Average closeness centrality 28.0514 sec
        Average betweenness centrality 36.1492 sec
        Pivots:
                Random pivot selection: 0.0439 sec
                Random degree pivot selection: 150.1215 sec
                Random page rank pivot selection: 74.9543 sec
                Degree pivot selection: 0.007 sec
                Page rank pivot selection: 2.1127 sec
                Page rank reverse pivot selection: 2.111 sec
                Page rank alternate pivot selection: 3.3904 sec

Barabasi-Albert Graph calculations
        Average closeness centrality 20.0752 sec
        Average betweenness centrality 18.1115 sec
        Pivots:
                Random pivot selection: 0.0259 sec
                Random degree pivot selection: 160.9023 sec
                Random page rank pivot selection: 76.3554 sec
                Degree pivot selection: 0.0051 sec
                Page rank pivot selection: 5.2688 sec
                Page rank reverse pivot selection: 5.1462 sec
                Page rank alternate pivot selection: 6.4302 sec

'''