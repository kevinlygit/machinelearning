import re
import operator
import random
import numpy as np

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
	p0Num=np.ones(numWords) # Initialize probabilities
	p1Num=np.ones(numWords)
	p0Denom=2.0
	p1Denom=2.0
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1Num+=trainMatrix[i] # Vector addition
			p1Denom+=sum(trainMatrix[i])
		else:
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
	p1Vect=np.log(p1Num/p1Denom) # change to log() Element-wise division
	p0Vect=np.log(p0Num/p0Denom) # change to log()
	return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1=sum(vec2Classify*p1Vec)+np.log(pClass1) # Element-wise multiplication
	p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
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
	p0V,p1V,pAb=trainNB0(np.array(trainMat),np.array(listClasses))
	testEntry=['love','my','dalmation']
	thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
	with open('output.out','w') as f:
		f.write(str(testEntry)+' classified as: '+str(classifyNB(thisDoc,p0V,p1V,pAb))+'\n')
	testEntry=['stupid','garbage']
	thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
	with open('output.out','a') as f:
		f.write(str(testEntry)+' classified as: '+str(classifyNB(thisDoc,p0V,p1V,pAb)))

def textParse(bigString):
	listOfTokens=re.split(r'[^\w\'(\d,\d)]+',bigString)
	listOfTokens2=[]
	for word in listOfTokens:
		if word=='':
			continue
		if ',' in word:
			word=word.replace(',','')
		word=word.lower()
		if len(word)>2:
			listOfTokens2.append(word)
	listOfTokens=listOfTokens2
	with open('output.out','a') as f:
		f.write(str(listOfTokens)+'\n')
	# return [tok.lower() for tok in listOfTokens if len(tok)>2]
	return listOfTokens

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
	"""
	for i in range(10):
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	"""
	testSet=random.sample(trainingSet,10) # Randomly create the training set
	for i in range(len(testSet)):
		trainingSet.remove(testSet[i])
	trainMat=[]
	trainClasses=[]
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount=0
	for docIndex in testSet: # Classify the test set
		wordVector=setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount+=1
	print('the error rate is:',errorCount/len(testSet))

def calcMostFreq(vocabList,fullText): # Calculates frequency of occurrence
	freqDict={}
	for token in vocabList:
		freqDict[token]=fullText.count(token)
	sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
	return sortedFreq[:30]

def getStopWord():
	with open('stopword.txt','r') as f:
		lineStr=f.read()
	stopWordList=re.split(r'[^\w\']+',lineStr)
	return stopWordList

def localWords(feed1,feed0):
	docList=[]
	classList=[]
	fullText=[]
	minLen=min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minLen):
		wordList=textParse(feed1['entries'][i]['summary']) # Accesses one feed at a time
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList=textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList=createVocabList(docList)
	top30Words=calcMostFreq(vocabList,fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList: # Removes most frequently occurring words
			vocabList.remove(pairW[0])
	trainingSet=list(range(2*minLen))
	testSet=[]
	"""
	for i in range(20):
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	"""
	testSet=random.sample(trainingSet,20)
	for i in range(len(testSet)):
		trainingSet.remove(testSet[i])
	trainMat=[]
	trainClasses=[]
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount=0
	for docIndex in testSet:
		wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount+=1
	print('the error rate is:',float(errorCount)/len(testSet))
	return vocabList,p0V,p1V

def getTopWords(ny,sf):
	vocabList,p0V,p1V=localWords(ny,sf)
	topNY=[]
	topSF=[]
	for i in range(len(p0V)):
		if p0V[i]>-6.0:
			topSF.append((vocabList[i],p0V[i]))
		if p1V[i]>-6.0:
			topNY.append((vocabList[i],p1V[i]))
	sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
	print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
	for item in sortedSF:
		print(item[0])
	sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
	print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
	for item in sortedNY:
		print(item[0])

def trainBayes():
	docList=[]
	classList=[]
	fullText=[]
	with open('trainset.txt','r') as f:
		lineListStr=f.readlines()
	docLen=len(lineListStr)
	for i in range(docLen):
		wordList=textParse(lineListStr[i]) # Accesses one feed at a time
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList=textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList=createVocabList(docList)
	top30Words=calcMostFreq(vocabList,fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList: # Removes most frequently occurring words
			vocabList.remove(pairW[0])
	trainingSet=list(range(2*minLen))
	testSet=[]
	"""
	for i in range(20):
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	"""
	testSet=random.sample(trainingSet,20)
	for i in range(len(testSet)):
		trainingSet.remove(testSet[i])
	trainMat=[]
	trainClasses=[]
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount=0
	for docIndex in testSet:
		wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount+=1
	print('the error rate is:',float(errorCount)/len(testSet))
	return vocabList,p0V,p1V

def trainBayes2():
	docList=[]
	classList=[]
	fullText=[]
	with open('trainset.txt','r') as f:
		lineListStr=f.readlines()
	docLen=len(lineListStr)
	for i in range(docLen):
		wordList=textParse(lineListStr[i]) # Accesses one feed at a time
		docList.append(wordList)
		fullText.extend(wordList)
	with open('trainlabel.txt','r') as f:
		lineStr=f.read()
	lineListStr=re.split(r'[\n]+',lineStr)
	lineListStr2=[]
	for word in lineListStr:
		if word=='':
			continue
		lineListStr2.append(int(word))
	classList=lineListStr2
	with open('output.out','a') as f:
		f.write(str(classList)+'\n')
	vocabList=createVocabList(docList)
	"""
	top30Words=calcMostFreq(vocabList,fullText)
	with open('output.out','a') as f:
		f.write(str(top30Words))
	for pairW in top30Words:
		if pairW[0] in vocabList: # Removes most frequently occurring words
			vocabList.remove(pairW[0])
	"""
	trainingSet=list(range(docLen))
	testSet=[]
	"""
	for i in range(20):
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	"""
	testSet=random.sample(trainingSet,0)
	for i in range(len(testSet)):
		trainingSet.remove(testSet[i])
	trainMat=[]
	trainClasses=[]
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount=0
	for docIndex in trainingSet:
		wordVector=setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount+=1
	print('the error rate is:',float(errorCount)/len(trainingSet))
	# return vocabList,p0V,p1V