import re
from numpy import *

def loadDataSet():
	postingList=[['my','dog','has','flea','problems','help','please'], \
	['maybe','not','take','him','to','dog','park','stupid'], \
	['my','dalmation','is','so','cute','I','love','him'], \
	['stop','posting','stupid','worthless','garbage'], \
	['mr','licks','ate','my','steak','how','to','stop','him'], \
	['quit','buying','worthless','dog','food','stupid']]
	classVec=[0,1,0,1,0,1] # 1 is abusive, 0 not
	return postingList,classVec

def createVocabList(dataSet):
	vocabSet=set([]) # Create an empty set
	for document in dataSet:
		vocabSet=vocabSet|set(document) # Create the union of two sets
	return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
	returnVec=[0]*len(vocabList) # Create a vector of all 0s
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]=1
		else:
			print('the word: %s is not in my Vocabulary!' % word)
	return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
	returnVec=[0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]+=1
	return returnVec

def trainNB0(trainMatrix,trainCategory):
	numTrainDocs=len(trainMatrix)
	numWords=len(trainMatrix[0])
	pAbusive=sum(trainCategory)/float(numTrainDocs)
	p0Num=ones(numWords) # Initialize probabilities
	p1Num=ones(numWords)
	p0Denom=2.0
	p1Denom=2.0
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1Num+=trainMatrix[i] # Vector addition
			p1Denom+=sum(trainMatrix[i])
		else:
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
	p1Vect=log(p1Num/p1Denom) # change to log() Element-wise division
	p0Vect=log(p0Num/p0Denom) # change to log()
	return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1=sum(vec2Classify*p1Vec)+log(pClass1) # Element-wise multiplication
	p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
	if p1>p0:
		return 1
	else:
		return 0

def testingNB():
	listOposts,listClasses=loadDataSet()
	myVocabList=createVocabList(listOposts)
	trainMat=[]
	for postinDoc in listOposts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
	testEntry=['love','my','dalmation']
	thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
	with open('../../../output.out','w') as f:
		f.write(str(testEntry)+' classified as: '+str(classifyNB(thisDoc,p0V,p1V,pAb))+'\n')
	testEntry=['stupid','garbage']
	thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
	with open('../../../output.out','a') as f:
		f.write(str(testEntry)+' classified as: '+str(classifyNB(thisDoc,p0V,p1V,pAb)))

def textParse(bigString):
	listOfTokens=re.split(r'\W+',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
	docList=[]
	classList=[]
	fullText=[]
	for i in range(1,26):
		wordList=textParse(open('email/spam/%d.txt' % i).read()) # Load and parse text files
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList=textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList=createVocabList(docList)
	trainingSet=list(range(50))
	testSet=[]
	testSet=random.sample(trainingSet,10) # Randomly create the training set
	trainMat=[]
	trainClasses=[]
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
	errorCount=0
	for docIndex in testSet: # Classify the test set
		wordVector=setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount+=1
	print('the error rate is:',errorCount/len(testSet))