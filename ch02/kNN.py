from numpy import *
import operator
from os import listdir

#创建记录
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

###knn分类算法
#inX,测试数据
#dataSet,训练样本数据
#labels，训练样本数据的分类标签
#k，表示取k个近邻数据
def classify0(inX, dataSet, labels, k): 
    dataSetSize = dataSet.shape[0] #获取数据集中记录的数目
    diffMat = tile(inX, (dataSetSize,1))-dataSet #测试数据与训练数据之差
    sqDiffMat = diffMat**2 #各个差的平方
    sqDistances = sqDiffMat.sum(axis=1) #同一记录的多个特征值差的平方之和
    distances = sqDistances**0.5 #求和之后再求平方根
    sortedDistIndicies = distances.argsort() #对平方根排序
    classCount={}
    for i in range(k):  #提取最小的k个值
        voteLabel = labels[sortedDistIndicies[i]] #获取记录的分类
        classCount[voteLabel] = classCount.get(voteLabel,0)+1 #对应分类标签的数目加1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0] #返回最多的那个分类标签

#解析文件，获取数据
#filename,文件名称
#resultMat存储数据的特征向量的矩阵
#classLabelVector存储数据的分类标签，就是已知的真实的分类结果
def file2matrix(filename):
    fr = open(filename) #打开文件
    arrayOfLines = fr.readlines() #读取文件内容  
    numberOfLines = len(arrayOfLines)  #获取记录数目，也就是行数
    resultMat = zeros((numberOfLines,3)) #产生 总行数*3 的矩阵
    classLabelVector = [] #用于记录真实的分类结果标签
    index = 0
    for line in arrayOfLines: #遍历文件内容，把对应内容存到矩阵和vector中
        line = line.strip()
        listFromLine = line.split('\t')
        resultMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return resultMat,classLabelVector

#归一化特征值
#dataSet数据集
#normDataSet 归一化后的数据集
#ranges 最大值与最小值的差
#minValues 最小值
def autoNorm(dataSet):
    minValues = dataSet.min(0) #每列的最小值
    maxValues = dataSet.max(0) #每列的最大值
    print('min value')
    print(minValues)
    print('max value')
    print(maxValues)
    ranges = maxValues - minValues  #最大值与最小值的差
    normDataSet = zeros(shape(dataSet))  #与dataSet维数相同的全0矩阵
    m = dataSet.shape[0] #dataSet的行数
    normDataSet = dataSet-tile(minValues,(m,1)) #将dataSet数据每个元素都减去最小值
    normDataSet = normDataSet/tile(ranges,(m,1)) #然后每个元素再除以（最大值与最小值之差）
    return normDataSet, ranges,minValues

#分类器对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10 #测试数据占总数据的比例为0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt') #读取文件存入数据结构
    normMat,ranges,minValues = autoNorm(datingDataMat) #归一特征值
    m = normMat.shape[0]  #总记录数
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs): #以前int(m*hoRatio)条记录作为测试数据，其他数据作为训练数据
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
            datingLabels[numTestVecs:m],3) 
        if(classifierResult != datingLabels[i]):errorCount +=1.0
    print('the total error rate is: %f' %(errorCount/float(numTestVecs)))

#约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing vidio games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat,datingLabels = file2matrix('datingTestSet.txt') #获取训练样本数据
    normMat,ranges,minValues = autoNorm(datingDataMat) #归一化特征
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0(inArr, datingDataMat, datingLabels, 3) #调用kNN算法
    print('You will probably like this person:',resultList[classifierResult-1])


#手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits') #文件夹下存储的是训练样本文件
    m = len(trainingFileList) #获取样本文件个数
    trainingMat = zeros((m,1024)) 
    #文件中是32*32的数据，转换成1*1024，以便应用前边的程序。将m个文件的存储到矩阵中
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr) #用文件名描述了对应文件中是什么数字
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr) #读入训练样本文件
    testFileList = listdir('testDigits') #文件夹下存储的是测试样样本文件
    errorCount = 0.0  #错误的数目
    mTest = len(testFileList) #测试样本的个数
    for i in range(mTest): #遍历每一个测试样本，进行测试
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) #根据训练样本，将测试样本分类，得到结果
        print('the classifier cam back with:%d, the real answer is:%d'\
            %(classifierResult, classNumStr))
        if(classifierResult != classNumStr):errorCount+=1.0
        print('\n the total number of errors is:%d' %errorCount)
        print('\n the total error rate is: %f' %(errorCount/float(mTest))) #错误率