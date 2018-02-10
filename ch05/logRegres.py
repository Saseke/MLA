#http://blog.csdn.net/lu597203933/article/details/38468303
#gradAscent函数中递归算法实现的解释建议阅读博客，内有详细的解释

from numpy import *
#读取文本文件
def loadDataSet():
    dataMat = [] #存储数据集
    labelMat = [] #存储类别标签
    fr = open('testSet.txt') #打开文件
    for line in fr.readlines():
        lineArr = line.strip().split() #去除每行的前后空格，以空格作为分隔符隔开
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #x0=1,x1,x2分别为取的数值
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX): #sigmoid函数
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) #把list转换为matrix
    labelMat = mat(classLabels).transpose() #balist转换为matrix，同时进行转置
    m,n = shape(dataMatrix) #数据的维度，这里是(n*3)
    alpha = 0.001  #定义步长，需要给定
    maxCycles = 500 #最大迭代次数
    weights = ones((n,1)) #存储迭代时的权重，初始化为1
    for k in range(maxCycles):  
        h = sigmoid(dataMatrix*weights)  #y=h(x)预测值
        error = (labelMat-h) #预测误差
        weights = weights+alpha*dataMatrix.transpose()*error #w=w+a*(y-h(x))*x,当只使用一个样本时迭代的公式
    return weights


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2] #0=w0x0+w1x1+...+wnxn,最佳拟合直线方程
    ax.plot(x,y)
    plt.xlabel('x1');plt.ylabel('x2')
    plt.show()

#随机梯度算法
def  atocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m): #遍历样本
        h = sigmoid(dataMatrix[i]*weights) #使用一个样本进行迭代
        error = classLabels[i] - h #计算误差
        weights = weights+alpha*dataMatrix[i]*error #只有一个数据集时，梯度迭代公式
    return mat(weights).transpose() #与书中不同，直接返回会出bug，因为plotBestFit的输入需要getA
#改进的梯度算法
#dataMatrix样本集，classLabels类别标签，numIter迭代次数，默认150次
def atocGradAscent1(dataMatrix,classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #alpha在每次迭代时都会调整，缓解了数据波动。虽然alpha会随着迭代次数不断减小，
            #但永远不会减小到0，这是因为始终还有0.01这个常数项，这就保证了多次迭代后，新数据
            #仍有一定的影响。
            alpha = 4/(1.0+j+i)+0.01  
            randIndex = int(random.uniform(0,len(dataIndex)))#随机选取样本，减少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights+alpha*error*dataMatrix[randIndex] #迭代，计算权值
            del(dataIndex[randIndex])
    return mat(weights).transpose()

#回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if(prob > 0.5): 
        return 1.0
    else :
        return 0.0
        
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(curLine[-1]))
        
    trainWeights = atocGradAscent1(array(trainingSet),trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec +=1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if(int(classifyVector(lineArr,trainWeights))!= int(currLine[-1])):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is : %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the \
        average error rate is: %f  " %(numTests, errorSum/float(numTests)))