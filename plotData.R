pkgTest <- function(x){
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE)
      if(!require(x,character.only = TRUE)) stop("Package not found")
  }
}

pkgTest('ggplot2')
library('ggplot2')

# # set to any directory you want, data must be there
# setwd('C:/Users/dru/Documents/[Dru]/Uni/SNACS/Centrality Paper/Graph-centrality-estimation-master')

# input dataframes
pivotValues = as.data.frame(read.csv('tables/pivotValues.csv'))
realValues = as.data.frame(read.csv('tables/realValues.csv'))

graphsList = as.character(realValues$graphType)

########### Plot pivots ###########

# iterate through all graphs
# plot betweenness & closeness pivot values for all graphs
for (g in c(1:length(graphsList))){
  
  titleMessage = paste(c(graphsList[g], 'Graph'), collapse = ' ')
  subtitleMessage = paste(c('Numbers of nodes:', prettyNum(realValues[g,5], big.mark = ',')),
                          collapse = '')
  
  # do twice, v = 1 closeness plot, v = 2 betweenness plot
  for (v in c(1:2)){
    # tweak plot parameters for each case
    if (v == 1){
      realValue = realValues[g,1]
      ylabVariable = 'closenessValue'
      ylabMessage = 'Closeness Value'
      fileName = paste(c('images/', realValues[g,5], '-', graphsList[g], '-Closeness.png'),
                       collapse = '')
    }else{
      realValue = realValues[g,2]
      ylabVariable = 'betweennessValue'
      ylabMessage = 'Betweenness Value'
      fileName = paste(c('images/', realValues[g,5], '-', graphsList[g], '-Betweenness.png'),
                       collapse = '')
    }
    
    # actual plotting happens here  
    thePlot = ggplot(subset(pivotValues, graphType == graphsList[g]),
                     aes_string('numberOfPivots', ylabVariable, 
                                color = 'pivotStrategy'),
                                log10 = 'y') +
                    geom_line(size = 0.8) +
                    geom_hline(yintercept = realValue, color = 'black') +
                    # log scales
                    scale_y_continuous(trans = 'log10') +
                    scale_x_continuous(trans = 'log10', labels = scales::comma) +
                  
                    labs(title = titleMessage,
                           subtitle = subtitleMessage,
                           x = 'Number of pivots',
                           y = ylabMessage,
                           color = 'Pivot \nselection \nstrategies') +
                      
                    theme(plot.title = element_text(hjust = 0.5),
                            plot.subtitle = element_text(hjust = 0.5),
                            text = element_text(size = 16),#, family='xkcd'),
                            legend.position = 'right')
    
    # # print to terminal or whatever
    # print(thePlot)

    # save to working directory
    ggsave(fileName, plot = thePlot, scale = 1.5, width = 12, height = 8, units = 'cm')
  }
}
 
# # print last plot
# print(thePlot)

########### Extract differences ###########

# list of strings with all pivot strategies names
pivotList = unique(pivotValues$pivotStrategy)

maxNumberOfPivots = max(realValues[,5])

# x axis: graph type
# y axis: pivot selection strategy
closenessDiffAll = matrix(NA, nrow = length(pivotList), ncol =  length(graphsList),
                          dimnames = list(pivotList, graphsList))

betweennessDiffAll = matrix(NA, nrow = length(pivotList), ncol =  length(graphsList),
                         dimnames = list(pivotList, graphsList))

# iterate through all graphs
for (graph in c(1:length(graphsList))){
  # iterate through all pivot strategies
  for (strategy in c(1:length(pivotList))){
    data = subset(pivotValues,
                  pivotStrategy == pivotList[strategy] &
                    graphType == graphsList[graph])
    
    closenessDiff = abs(data$closenessValue - realValues[graph,1])
    closenessDiffAll[strategy, graph] = round(mean(closenessDiff, na.rm = T)*10^4, 2)
    
    betweennessDiff = abs(data$betweennessValue - realValues[graph,2])
    betweennessDiffAll[strategy, graph] = round(mean(betweennessDiff, na.rm = T)*10^4, 2)
  }
}