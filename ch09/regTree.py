
#树节点结构
class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

from numpy import *
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #将每行映射为浮点数
        dataMat.append(fltLine)
    return dataMat
def binSplitDataSet(dataSet, feature, value): #要将数据局分为两部分
    #mat0 = dataSet[nonzero(dataSet[:,feature]>value)[0],:][0]
    #mat1 = dataSet[nonzero(dataSet[:,feature]<=value)[0],:][0]
    #书中有误，应该为下式
    mat0 = dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    #nonzero函数，用于获得数组元素值不为0的元素的下表索引 
    mat1 = dataSet[nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0, mat1



#回归树的切分函数
def regLeaf(dataSet): #负责生成叶节点，当要退出递归时，调用得到叶节点的模型，在回归树种就是目标变量的均值
    return mean(dataSet[:,-1])
def regErr(dataSet): #误差估计好似，给定数据上计算目标变量的平方误差
    return var(dataSet[:,-1])*shape(dataSet)[0] #均方差乘以样本个数
def chooseBestSplit(dataSet, leafType=regLeaf,errType=regErr,ops=(1,4)):
    #tolS与tolN是用户指定的参数，用于控制函数的停止时机
    tolS=ops[0] #容许的误差下降值
    tolN=ops[1] #切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #若剩余特征数目为1，不需切分，退出
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet) #计算当前误差
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1): #遍历所有特征属性
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]): #遍历单一特征下的所有的特征属性值
            mat0,mat1 = binSplitDataSet(dataSet, featIndex, splitVal) #按照属性和属性值分割
            if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN): #如果分割的样本小于4，则跳出循环
                continue
            newS = errType(mat0)+errType(mat1) #计算分割后的误差
            if(newS < bestS): #若比最佳误差还小，则赋值。
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    if(S-bestS)<tolS: #原误差与分割后的最佳误差的差值 小于 允许的误差下降值
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0]<tolN): #最小样本数小于4，则不再分割
        return None,leafType(dataSet)
    return bestIndex,bestValue #如果不满足终止条件，那么返回切分特征和特征值
#创建树，首先将数据集分为两部分，切分由chooseBestSplit完成。如果满足停止条件
#如果构建的是回归树，该模型是一个常数，如果是模型树，其模型是一个线性方程；
#而如果不满足停止条件，choosBestSplit()将创建一个新的python字典并将数据集分成两份
#在这两份数据集上将分别继续地柜调用createTree函数
#leafType:对创建叶节点的函数的引用
#errType是总方差计算函数的引用
#ops是一个用户定义的参数构成的元组，用以完成树的构建
def createTree(dataSet, leafType=regLeaf, errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops) #选择最佳的分割方式
    if feat == None: #满足停止条件，退出，否则递归调用
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet, feat, val) #分割数据
    retTree['left'] = createTree(lSet, leafType, errType, ops) #左子树递归分割
    retTree['right'] = createTree(rSet, leafType, errType, ops) #右子树递归分割
    return retTree



#回归树剪枝函数
def isTree(obj): #判断是不是一棵树
    return (type(obj).__name__=='dict')
def getMean(tree): #获取树的平均值
    if isTree(tree['right']): #右子树平均值
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): #左子树平均值
        tree['left'] = getMean(tree['left'])
    return(tree['left']+tree['right'])/2.0 #左右子树再进行平均

def prune(tree, testData):
    if(shape(testData)[0]==0):
        return getMean(tree) #没有测试数据，对树进行塌陷处理，就是返回树的平均值
    if(isTree(tree['right'])) or isTree(tree['left']): #判断左右子树是不是树，如果是树，就分割数据
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']): #如果左子树是树，剪枝处理
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']): #如果右子树是树，剪枝处理
        tree['right'] = prune(tree['right'],rSet)
    #左右子树剪枝处理后，若左右子树均不是树，就可以进行合并。
    #合并做法：对合并前后的误差进行比较，若合并后的误差比不合并的误差小，就进行合并操作，否则不合并
    if not isTree(tree['left']) and not isTree(tree['right']): 
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1]-tree['left'],2))+\
            sum(power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    return tree

#模型树的叶节点生成函数
def linearSolve(dataSet):  #将数据集格式化成目标变量y和自变量X，执行了简单的线性回归
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError("this matrix is singular, cannot do inverse,\n\
            try increasing the second value of ops")
    ws = xTx.T*(X.T*Y)
    return ws,X,Y
def modelLeaf(dataSet):#当数据不再需要切分的时候，它负责生成叶节点的模型
    ws,X,Y = linearSolve(dataSet)
    return ws
def modelErr(dataSet): #计算误差
    ws,X,Y=linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y-yHat,2))

#用树回归进行预测的代码
def regTreeEval(model,inDat):
    return float(model)
def modelTreeEval(model,inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)
#treeForceCast
#对于输入的单个数据点或行向量，函数会返回一个浮点值。在给定树结构的情况下，
#对于单个数据点，该函数会给出一个预测值。调用该函数需要指定树的类型，以便
#在叶节点上能够调用合适的模型。参数modelEval是对叶节点数据进行预测的函数的引用
#treeForeCast自顶向下遍历整棵树，直到命中叶节点为止。一旦达到叶节点，
#它就会在输入数据上调用modleEval函数，而该函数的默认值是regTreeEval
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
             return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'],inData)
def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yMat = mat(zeros((m,1)))
    for i in range(m):
        yMat[i,0] = treeForeCast(tree, mat(testData[i]),modelEval)
    return yMat