
from numpy import *
#创建一个简单的数据集
def loadSimpleData():
    dataMat = matrix(([1.,2.1],
            [2.,1.1],
            [1.3,1.],
            [1.,1.],
            [2.,1.]))
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels


def stumpClassify(dataMatrix,dimen, threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1)) #先将返回结果初始化为1的列向量
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal]= -1.0  #小于，其值赋为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0  #大于，其值赋为-1
    return retArray

def buildStump(dataArr, classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n): #遍历特征值
        rangeMin = dataMatrix[:,i].min() #特征值的最小值
        rangeMax = dataMatrix[:,i].max() #特征值的最大值
        stepSize = (rangeMax-rangeMin)/numSteps  #遍历特征值时的步长
        for j in range(-1,int(numSteps)+1): #根据步长遍历特征值
            for inequal in ['lt','gt']: #不等式的符号
                threshVal = (rangeMin+float(j)*stepSize) #每次分类的阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) #根据数据集，特征，阈值，不等号分类
                errArr = mat(ones((m,1))) #error初始化为1
                errArr[predictedVals == labelMat] = 0 #相等的err设为9
                weightedError = D.T*errArr #计算加权错误率
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i  #属性列标号，就是记录哪一列作为分类
                    bestStump['thresh'] = threshVal #阈值
                    bestStump['ineq'] = inequal #是大于还是小于
    return bestStump,minError,bestClassEst  #bestStump返回的分类器信息，minErr错误率，bestClassEst预测值

def adaBoostTrainDS(dataArr, classLabels, numIt = 40): #numIt迭代次数，需要用户唯一指定的参数
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16))) #计算alpha权重，分类器结果统计时的权重
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:",classEst.T) #预测信息
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #这里只有1和-1，也就表示正确时为-alpha,错误时为alpha
        D = multiply(D,exp(expon))
        D = D/D.sum() #求权重向量D
        aggClassEst += alpha*classEst #类别估计值
        print("aggClassEst:",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) !=mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m #总的错误率
        print("total error:",errorRate,"\n")
        if(errorRate == 0.0): #若错误率为0，直接跳出迭代
            break;
    #return weakClassArr #返回结果包含了每一次迭代的分类器信息
    #下面的return语句在测试plot roc时使用
    return weakClassArr,aggClassEst
#以上已完成了adaBoost分类的大部分代码，下面的函数是如何应用
#每一个弱分类器，得到结果，然后根据alpha加权求和
#adaBoost分类函数
def adaClassify(dataToClass, classifierArr): #数据集，弱分类器信息
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0] #有多少数据记录
    aggClassEst = mat(zeros((m,1))) #各记录预测值
    for i in range(len(classifierArr)): #遍历弱分类器
        #调用用弱分类器，得到预测值
        classEst = stumpClassify(dataMat,classifierArr[i]['dim'], \
            classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst #加权求和
        print (aggClassEst) #预测值
    return sign(aggClassEst) #得到最终的类别信息

#使用adaBoost分类函数的步骤
#loadSimpleData,获取数据集和类别标签
#adaBoostTrainDS, 训练算法，得到分类器信息
#adaClassify, 分类，得到结果

#自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) #获取特征数目
    dataMat = []
    labelMat = []
    fr = open(fileName) #打开文件
    for content in fr.readlines(): #遍历每一行
        lineArr = []
        curLine = content.strip().split('\t') #按tab键分割
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)  #特征集合
        labelMat.append(float(curLine[-1])) #类别标签
    return dataMat,labelMat #返回数据


#进行测试函数。在当前目录下，保存了马疝病数据集
#horseColicTraining2.txt, horseColicTest2.txt
#下面的函数列出了使用adaBoostClassify函数的应用模板
def test():
    dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
    #训练算法，得到分类器信息
    classifierArr = adaBoostTrainDS(dataArr, labelArr, 10) #numItr=10,最多迭代10次，也就是分类器数目最多为10

    dataArr,labelArr = loadDataSet('horseColicTest2.txt')
    #分类
    prediction10 = adaClassify(dataArr, classifierArr)
    num = len(dataArr)
    errorArr = mat(ones((num,1)))
    errorNum = errorArr[prediction10 != mat(labelArr).T].sum()
    print("the Error numbers %d , rate %f" % (int(errorNum), float(errorNum)/num))  


#ROC曲线的绘制以及AUC计算函数
def plotROC(predStrengths,classLabels):
    import matplotlib.pyplot as plt
    cur =[1.0,1.0]
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0) #所有正例的数目
    yStep = 1/float(numPosClas) #y轴步进
    xStep = 1/float(len(classLabels)-numPosClas) #x轴步进
    sortedIndicies = predStrengths.argsort() #对分类器预测强度排序
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0 : #每遇到一个标签为1.0的类，则沿y轴的方向下降一个步长，降低真阳率
            delX = 0
            delY = yStep
        else: #遇到其他类的标签，则在X轴方向倒退一个步长(假阴率方向)
            delX =xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX], [cur[1],cur[1]-delY],c='b')
        cur = [cur[0]-delX, cur[1]-delY]
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    plt.title('ROC curve for adaboost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area under the curve is:",ySum*xStep)

        