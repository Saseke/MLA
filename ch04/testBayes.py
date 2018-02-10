import bayes
from numpy import *

def testNB():
    listOPosts,listClasses = bayes.loadDataSet() #加载数据
    myVocabList = bayes.createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc)) 
    p0V,p1V,pAb= bayes.trainNB0(trainMat,listClasses)

    resultLabel = {0:'Not garbage',1:'Garbage'}
    testEntry = ['love','my','dalmation']
    thisDoc = array(bayes.setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',resultLabel[bayes.classifyNB(thisDoc,p0V,p1V,pAb)])

    testEntry = ['stupid','garbage']
    thisDoc = array(bayes.setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',resultLabel[bayes.classifyNB(thisDoc,p0V,p1V,pAb)])