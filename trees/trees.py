import math
import operator
import matplotlib.pyplot as plt
import pickle

def calcShannonEnt(dataSet):
	numEntries=len(dataSet)
	labelCounts={}
	for featVec in dataSet: # Create dictionary of all possible classes
		currentLabel=featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	shannonEnt=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/numEntries
		shannonEnt-=prob*math.log(prob,2) # Logarithm base 2
	return shannonEnt

def createDataSet():
	dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels=['no surfacing','flippers']
	return dataSet,labels

def splitDataSet(dataSet,axis,value):
	retDataSet=[] # Create separate list
	for featVec in dataSet:
		if featVec[axis]==value:
			reducedFeatVec=featVec[:axis] # Cut out the feature split on
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1
	baseEntropy=calcShannonEnt(dataSet)
	bestInfoGain=0.0
	bestFeature=-1
	for i in range(numFeatures):
		featList=[example[i] for example in dataSet] # Create unique list of class labels
		uniqueVals=set(featList)
		newEntropy=0.0
		for value in uniqueVals: # Calculate entropy for each split
			subDataSet=splitDataSet(dataSet,i,value)
			prob=len(subDataSet)/float(len(dataSet))
			newEntropy+=prob*calcShannonEnt(subDataSet)
		infoGain=baseEntropy-newEntropy
		if (infoGain>bestInfoGain):
			bestInfoGain=infoGain # Find the best information gain
			bestFeature=i
	return bestFeature

def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote]=0
		classCount[vote]+=1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def createTree(dataSet,labels):
	classList=[example[-1] for example in dataSet]
	if classList.count(classList[0])==len(classList): # Stop when all classes are equal
		return classList[0]
	if len(dataSet[0])==1: # When no more features, return majority
		return majorityCnt(classList)
	bestFeat=chooseBestFeatureToSplit(dataSet)
	bestFeatLabel=labels[bestFeat]
	myTree={bestFeatLabel:{}}
	del(labels[bestFeat]) # Get list of unique values
	featValues=[example[bestFeat] for example in dataSet]
	uniqueVals=set(featValues)
	for value in uniqueVals:
		subLabels=labels[:]
		myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree

def classify(inputTree,featLabels,testVec):
	firstStr=list(inputTree.keys())[0]
	secondDict=inputTree[firstStr]
	featIndex=featLabels.index(firstStr) # Translate label string to index
	for key in secondDict.keys():
		if testVec[featIndex]==key:
			if type(secondDict[key]).__name__=='dict':
				classLabel=classify(secondDict[key],featLabels,testVec)
			else:
				classLabel=secondDict[key]
	return classLabel

def storeTree(inputTree,filename):
	with open(filename,'wb') as fwb:
		pickle.dump(inputTree,fwb)

def grabTree(filename):
	with open(filename,'rb') as frb:
		return pickle.load(frb)