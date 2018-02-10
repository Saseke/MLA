from numpy import *

#创建实验样本，样本也可以读取文本获得
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1] #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

#创建词汇表，不重复
def createVocabList(dataSet):
    vocabSet = set([])  #set构造函数会返回一个不重复的词表
    for doc in dataSet:
        vocabSet = vocabSet|set(doc)  #求并集
    return list(vocabSet)

#创建词汇表vocabList后，读取测试文件inputSet内容，记录在该文件中词汇的出现情况，并返回记录结果
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList) #与词汇表等长的列表，用于记录结果
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =1 #若出现，将列表对应值设置为1
        else:
            print("the word %s is not in my vocabulary" % word)
    return returnVec #返回结果

#朴素贝叶斯分类器训练函数
#trainMatrix，输入，是经过setOfWords2Vec获得的的各样本文档中词汇的出现情况的向量
#trainCategory，输入，是各样本文档的类别标签向量
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) #文档总数
    numWords = len(trainMatrix[0]) #词汇总数目
    pAbusive = sum(trainCategory)/float(numTrainDocs) #任意文档为侮辱性文档的概率
    '''
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0  #p0时所有文档的总词汇数
    p1Denom = 0.0  #p1时所有文档的总词汇数
    '''
    '''
    在分类文档时，需要计算多个概率的乘积以获得文档属于某个类别的概率。那么
    如果其中一个概率值为0，就会导致最后的乘积也变为0.为降低该影响，将所有词
    的出现数初始化为1，对应的分母初始化为2
    '''
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0  #p0时所有文档的总词汇数
    p1Denom = 2.0  #p1时所有文档的总词汇数
    for i in range(numTrainDocs): #遍历所有文档
        if trainCategory[i] == 1: #如果是侮辱性文档
            p1Num += trainMatrix[i]; #文档中出现某词汇，就将对应的num加1
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i];
            p0Denom += sum(trainMatrix[i])
    '''
    p1Vect = p1Num/p1Denom #记录了在侮辱性文档中，各个词汇出现概率
    p0Vect = p0Num/p0Denom #记录了在非侮辱性文档中，各个词汇出现的概率
    '''
    '''
    求联合概率时，大部分银子都会非常小，所以程序会向下溢出或者
    得到不正确的答案，为解决该问题，对乘积取自然对数。
    因为ln(a*b)=ln(a)+ln(b)
    '''
    p1Vect = log(p1Num/p1Denom) #记录了在侮辱性文档中，各个词汇出现概率
    p0Vect = log(p0Num/p0Denom) #记录了在非侮辱性文档中，各个词汇出现的概率
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0V,p1V,pClass1):
    p1 = sum(vec2Classify*p1V)+log(pClass1)
    p0 = sum(vec2Classify*p0V)+log(1.0-pClass1) #案例中只有两类，故可以这么求
    if(p1>p0):
        return 1
    else:
        return 0
#setOfWords2Vec函数中只是记录了每个词汇是否出现，值只能取1或0
#为解决该问题，词袋模型，其可以记录每个词汇出现的次数
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] +=1
    return returnVec