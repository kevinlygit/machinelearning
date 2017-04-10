import math
import operator
import matplotlib.pyplot as plt
import pickle
import trees
import treeplotter

myDat,labels=trees.createDataSet()
myTree=treeplotter.retrieveTree(0)
trees.storeTree(myTree,'classifierstorage.txt')
ans=trees.grabTree('classifierstorage.txt')

with open('output.out','w') as f:
	f.write(str(ans))