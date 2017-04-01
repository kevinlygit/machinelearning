import os
import operator
import numpy as np

def createDataSet():
	group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	dataSetSize=dataSet.shape[0]
	diffMat=np.tile(inX,(dataSetSize,1))-dataSet # Distance calculation
	sqDiffMat=diffMat**2
	sqDistances=sqDiffMat.sum(axis=1)
	distances=sqDistances**0.5
	sortedDistIndicies=distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndicies[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 # Voting with lowest k distances
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) # Sort dictionary
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr=open(filename)
	arrayOLines=fr.readlines()
	numberOfLines=len(arrayOLines) # Get number of lines in file
	returnMat=np.zeros((numberOfLines,3)) # Create NumPy matrix to return
	classLabelVector=[]
	index=0
	for line in arrayOLines: # Parse line to a list
		line=line.strip()
		listFromLine=line.split('\t')
		returnMat[index,:]=listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index+=1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	normDataSet=zeros(shape(dataSet))
	m=dataSet.shape[0]
	normDataSet=dataSet-np.tile(minVals,(m,1))
	normDataSet=normDataSet/tile(ranges,(m,1)) # Element-wise division
	return normDataSet,ranges,minVals

def datingClassTest():
	hoRario=0.10
	datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRario)
	errorCount=0.0
	for i in range(numTestVecs):
		classifierResult=classify0(normMat[i,:]normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print('the classifier came back with: %d, the real answer is: %d' % (classifierResult,datingLabels[i]))
		if (classifierResult!=datingLabels[i]):
			errorCount+=1.0
	print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def classifyPerson():
	resultList=['not at all','in small doses','in large doses']
	percentTats=float(raw_input('percentage of time spent playing video games?'))
	ffMiles=float(raw_input('frequent flier miles earned per year?'))
	iceCream=float(raw_input('liters of ice cream consumed per year?'))
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	inArr=np.array([ffMiles,percentTats,iceCream])
	classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print('You will probably like this person: ',resultList[classifierResult-1])

def img2vector(filename):
	returnVect=zeros((1,1024))
	fr=open(filename)
	for i in range(32):
		lineStr=fr.readline()
		for j in range(32):
			returnVect[0,32*i+j]=int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels=[]
	trainingFileList=os.listdir('trainingDigits') # Get contents of directory
	m=len(trainingFileList)
	trainingMat=zeros((m,1024))
	for i in range(m):
		fileNameStr=trainingFileList[i] # Process class num from filename
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
	testFileList=os.listdir('testDigits')
	errorCount=0.0
	mTest=len(testFileList)
	for i in range(mTest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
		classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print('the classifier came back with: %d, the real answer is: %d' % (classifierResult,classNumStr))
		if (classifierResult!=classNumStr):
			errorCount+=1.0
	print('\nthe total number of errors is: %d' % errorCount)
	print('\nthe total error rate is: %f' % (errorCount/float(mTest)))