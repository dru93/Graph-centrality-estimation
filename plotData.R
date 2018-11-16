# messing around with fonts
library(extrafont)
library(ggplot2)

# # uncomment block if for first runtime
# xkcdFontURL = "http://simonsoftware.se/other/xkcd.ttf"
# download.file(xkcdFontURL,dest="xkcd.ttf",mode="wb")
# # takes a looot of time 10mins or smthing
# font_import()

# set to any directory you want, data must be there
setwd('C:/Users/dru/Documents/[Dru]/Uni/SNACS/Centrality Paper/Graph-centrality-estimation-master')

# input dataframes
pivotValues = read.csv('pivotValues-100n.csv')
realValues = read.csv('realValues-100n.csv')
# rename pivotStrategies 
pivotValues[,4] = substring(pivotValues[,4], first = 1, last = 6)

# display data just for ease
View(pivotValues)
View(realValues)

graphsList = c('Erdos-Renyi', 'Watts-Strogatz', 'Barabasi-Albert')

# iterate through all graphs
# plot betweenness & closeness pivot values for all graphs
for (g in c(1:length(graphsList))){
  
  titleMessage = paste(c(graphsList[g], 'Graph'), collapse = ' ')
  subtitleMessage = paste(c('Numbers of nodes:', realValues[g,5]), collapse = ' ')
  
  # do twice, v = 1 closeness plot, v = 2 betweenness plot
  for (v in c(1:2)){
    # tweak plot parameters for each case
    if (v == 1){
      ylabMessage = 'Closeness Value'
      fileName = paste(c(graphsList[g], realValues[g,5], 'Closeness.png'), collapse = '')
      realValue = realValues[g,1]
      ylabVariable = 'closenessValue'
    }else{
      ylabMessage = 'Betweenness Value'
      fileName = paste(c(graphsList[g], realValues[g,5], 'Betweenness.png'), collapse = '')
      realValue = realValues[g,2]
      ylabVariable = 'betweennessValue'
    }
    
    # actual plotting happens here  
    thePlot = ggplot(subset(pivotValues, graphType == graphsList[g]),
                     aes_string('numberOfPivots', ylabVariable, 
                                color = 'pivotStrategy')
    ) +
      geom_line(size = 1) +
      geom_hline(yintercept = realValue, color = 'black') +
      
      labs(title = titleMessage,
           subtitle = subtitleMessage,
           x = 'Number of pivots',
           y = ylabMessage,
           color = 'Pivot \nselection \nstrategies') +
      
      theme(plot.title = element_text(hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5),
            # comment below if fonts didn't really work out for you
            text = element_text(size=16, family='xkcd'),
            legend.position = 'right')
    
    # print to terminal or whatever
    print(thePlot)
    
    # save to working directory
    ggsave(fileName, plot = thePlot, scale = 1.5, width = 12, height = 8, units = 'cm')
  }
}

# print last plot
print(thePlot)