from numpy import *

def loadDataSet(fileName):#导入数据
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat  #返回xArr和yArr
#标准线性回归函数
def standRegres(xArr, yArr):  #总体运算方程：w=(x转置*x)的逆*x的逆*y
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0: #判断x是否可逆
        print('this matrix is sigular,cannot do inverse')
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

#利用standRegres得到回归系数w后，就可以根据测量值x得到预测值y了。
#y=x的转置*w

#局部加权线性回归函数
def lwlr(testPoint, xArr,yArr,k=1.0): #k由用户指定，控制衰减的速度
    xmat = mat(xArr)
    ymat = mat(yArr).T
    m = shape(xmat)[0] #训练样本的数目
    weight = mat(eye(m)) #每个样本对应一个权重，m阶单位矩阵
    for j in range(m): #遍历每条样本，得到权重矩阵w
        diffMat = testPoint - xmat[j,:]
        weight[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) #高斯核
    xTx = xmat.T*(weight*xmat) 
    if linalg.det(xTx) == 0.0: #判断是否可逆
        print('this matrix is sigular,cannot do inverse')
        return
    ws = xTx.I*(xmat.T*(weight*ymat)) #求得ws
    return testPoint*ws #返回预测值

#lwlr测试函数
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yMat = zeros(m)
    for i in range(m):
        yMat[i] = lwlr(testArr[i],xArr,yArr,k) #每个样本集都需要计算一次数据集
    return yMat

#预测鲍鱼的年龄
def rssError(yArr,yHatArr):#该函数用于返回预测误差的大小
    return ((yArr-yHatArr)**2).sum()
def baoyuTest():
    abX,abY=loadDataSet('abalone.txt')
    #训练数据
    print("trainingData:")
    k = 0.1
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], k)
    print("k=%f, error=%f" %(k, rssError(abY[0:99], yHat01)))
    k = 1
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], k)
    print("k=%f, error=%f" %(k, rssError(abY[0:99], yHat1)))
    k = 10
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], k)
    print("k=%f, error=%f" %(k, rssError(abY[0:99], yHat10)))

    #测试数据
    print("testData:")
    k = 0.1
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], k)
    print("k=%f, error=%f" %(k, rssError(abY[0:99], yHat01)))
    k = 1
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], k)
    print("k=%f, error=%f" %(k, rssError(abY[0:99], yHat1)))
    k = 10
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], k)
    print("k=%f, error=%f" %(k, rssError(abY[0:99], yHat10)))

#岭回归函数
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx+eye(shape(xMat)[1])*lam  #岭回归的核心，加了个对角矩阵
    if linalg.det(denom)==0.0:
        print("the matrix is singular, cannot do inverse")
        return
    ws = denom.I*(xMat.T*yMat) #根据权重公式计算
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr) #测试数据
    yMat = mat(yArr).T #测试的值
    yMean = mean(yMat,0) #求均值
    yMat = yMat - yMean 
    xMeans = mean(xMat,0) #求均值
    xVar = var(xMat,0) #x的协方差
    xMat = (xMat-xMeans)/xVar #数据标准化
    numTestPts = 30 #测试的lamda的范围
    wMat = zeros((numTestPts,shape(xMat)[1])) #存储不同lamda得到的权重值
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10)) #岭回归函数
        wMat[i,:] = ws.T #每一行对应一个lamda
    return wMat

    
#正则化函数
def regularize(xMat):#按列正则化
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #均值
    inVar = var(inMat,0)      #协方差
    inMat = (inMat - inMeans)/inVar #
    return inMat

#前向逐步线性回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean  #数据标准化
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1)) #n个特征对应的权向量，初始化为1
    wsTest = ws.copy() 
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n): #对每个特征值都执行了两次for循环，分别计算增加或减少该特征对误差的影响
            for sign in [-1,1]: #做加1个eps或减1个eps处理
                wsTest =ws.copy() 
                wsTest[j] += eps*sign 
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A) #返回预测误差的平方和
                if rssE < lowestError: #如果rssE小于最小误差，则迭代
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat




#使用google api获取数据，返回json格式的文件，python提供了json的解读模块
#获取数据这部分是直接拷贝书中的源码，没有分析
from time import sleep
import json
import urllib.request
def searchForSet(retX, retY, setNum, yr, numPce, origPrc): 
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print ('problem with item %d' % i)
    
def setDataCollect(retX, retY): #调用该函数时。连接失败，考虑监管问题
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr,yArr,numVal=10): #交叉验证
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]  #分为训练集和测试集
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]]) #训练集
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]]) #测试集
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX) #数据标准化
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)