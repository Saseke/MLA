from math import log
import operator

#自己创建数据
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
#计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #数据集数目
    labelsCounts = {} #记录分类标签以及数据集中各分类的样本数目
    for featVec in dataSet:
        currentLabel = featVec[-1] #创建一个数据字典，以最后一列作为键值
        if currentLabel not in labelsCounts.keys(): #若该分类不在labelsCounts中，则添加一项
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] +=1 #对应分类的样本数目加1
    shannonEnt = 0.0
    for key in labelsCounts: #计算各分类的香农熵
        prob = float(labelsCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

#按照给定特征划分数据集
#dataSet:代划分的数据集
#axis：划分数据集的特征
#value：特征值
#功能：将符合特征值的数据集抽取出来
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #这两步操作是将对应特征的那一列去掉，生成新的数据集
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最佳的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1  #要求数据集的最后一列或最后一个元素为类别标签
    baseEntropy = calcShannonEnt(dataSet) #计算香农熵
    bestInfoGain= 0.0 #最佳的增益
    bestFeature=-1 #最佳的划分属性类别
    for i in range(numFeatures): #访问所有的特征列属性
        featList = [example[i] for example in dataSet] #取一列特征值 
        uniqueValues = set(featList) #将list类型转换为set集合（set是无序且不重复的元素集合）
        newEntropy = 0.0
        for value in uniqueValues: #遍历当前特征中所有唯一的属性值，对每个特征划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value) 
            prob = len(subDataSet)/float(len(dataSet)) #求得数据集新熵值
            newEntropy +=prob*calcShannonEnt(subDataSet) #求得熵的综合
        infoGain = baseEntropy-newEntropy #信息增益
        if(infoGain>bestInfoGain): #比较所有特征中的增益，返回最好特征划分的索引值
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#如果数据集已处理了所有属性，但是类标签依然不是唯一的，
#此时我们需要决定如何定义该叶子节点，这种情况下，
#我们通常会采用多数表决的方法决定该叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),\
        key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建树的函数代码
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] #最后一列为类别标签
    #如果类别完全相同，则停止继续划分。count函数用于统计classList[0]出现的次数
    if(classList.count(classList[0]) == len(classList)): 
        return classList[0]
    if(len(dataSet[0])==1): #遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择最佳属性进行数据划分
    bestFeatLabel = labels[bestFeat] #最佳属性的label
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet] #得到列表包含的所有属性值
    uniqueValues = set(featValues) #转换为无序不重复的set集合
    for value in uniqueValues: #
        subLabels = labels[:] #去除bestFeatLabel之后的label
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
            (dataSet, bestFeat, value), subLabels)
    return myTree

#使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStrSides = list(inputTree.keys()) #获取标签字符串
    firstStr = firstStrSides[0]
    secondDict = inputTree[firstStr] #标签对应的子数据集
    featIndex = featLabels.index(firstStr) #将标签字符串转换为索引
    for key in secondDict.keys(): #遍历整棵树
        if testVec[featIndex] == key: #比较testVec中值与树节点的值，若达到叶子节点，则返回当前节点的分类标签
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#构造决策树很耗时，浪费时间，可先使用pickle模块存储已购造好的决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)