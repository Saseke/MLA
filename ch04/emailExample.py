#使用朴素贝叶斯实现垃圾邮件过滤案例
#该程序使用了bayes.py模块

import bayes
import random
from numpy import*
#解析文本文件，并将解析的结果返回
def textParse(textString):
    import re
    #将文本分隔开，这里分隔符采用通配符，是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W*',textString)
    #tok.lower,将字母统一转换为小写
    #len(tok)>2,将字符串长度小于3的过滤掉
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    #fullText = [] #没起作用

    for i in range(1,26): #此案例中样本集名为1.txt~25.txt
        wordList = textParse(open('email/spam/%d.txt' % i).read()) #解析邮件，分隔成一个个词汇
        docList.append(wordList)  #将样本内容存储到docList中
        #fullText.extend(wordList)
        classList.append(1) #spam下对应的类别标签设置为1
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        #fullText.extend(wordList)
        classList.append(0) #ham下对应的类别标签设置为0
    vocabList = bayes.createVocabList(docList) #通过docList获取全部的词汇表
    trainingSet = list(range(50)) #此处共50个案例，与classList长度对应
    
    testSet = [] #存储测试样本集
    for i in list(range(10)):
        randIndex = int(random.uniform(0,len(trainingSet))) #随机提取样本作为测试样本
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) #把测试样本从训练样本中剔除
    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:#遍历训练样本集
        trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex])) #获取样本中使用词汇情况向量
        trainClasses.append(classList[docIndex])  #获取当前样本的类别标签
    p0V,p1V,pSpam = bayes.trainNB0(array(trainMat), array(trainClasses)) #训练算法，得到概率
    errorCount = 0

    for docIndex in testSet: #遍历测试样本集
        wordVector=bayes.setOfWords2Vec(vocabList, docList[docIndex])
        resultFlag = bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) #使用分类函数进行分类
        if(resultFlag != classList[docIndex]): #如果得到结果不正确，则错误数加上1
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))