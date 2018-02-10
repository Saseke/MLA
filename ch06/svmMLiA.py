from numpy import *

def loadDataSet(fileName): #加载数据
    dataMat = []
    labelMat = []
    fr = open(fileName) #打开文档
    for line in fr.readlines(): #按行读取
        lineArr = line.strip().split('\t') #tab键分割
        dataMat.append([float(lineArr[0]), float(lineArr[1])])#前两列为数据
        labelMat.append(float(lineArr[2]))  #类别标签，要么为1，要么为-1
    return dataMat,labelMat

def selectJrand(i,m):#在0到m-1内随机选择一个不等于i的数
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clibAlpha(aj,H,L): 
    if aj>H: #大于H，等于H
        aj = H
    elif aj<L: #小于L，等于L
        aj = L
    return aj #返回aj

    
#简化版SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter<maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*\
                (dataMatrix*dataMatrix[i,:].T))+b
            Ei = fXi - float(labelMat[i])
            if((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or \
                ((labelMat[i]*Ei>toler) and alphas[i]>0):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas,labelMat).T*\
                (dataMatrix*dataMatrix[j,:].T))+b
                Ej =fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L==H: 
                    print('L==H')
                    continue
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    dataMatrix[i,:]*dataMatrix[i,:]-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clibAlpha(alphas[j], H, L)
                if (abs(alphas[j]-alphaJold)<0.0001):
                    print("j not moving enough")
                    continue
                alpha[i] += labelMat[j]*labelMat[i]*\
                    (alphaIold-alphas[j])
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0<alphas[j]) and (C>alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged +=1
                print("iter:%d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))
        if(alphaPairsChanged == 0):
            iter +=1
        else:
            iter = 0
        print ("iteration number: %d" %iter)
    return b,alphas

#完整的platt SMO的支持函数
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C =C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)
'''
def calcEk(oS,k):
    fXk = float(multiply(oS.alphas, os.labelMat).T*\
        (os.X[k,:].T))+oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
'''
def calcEk(oS,k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T*\
        (oS.K[:,k]))+oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
def selectJ(i, oS, Ei):
    maxK = -1;
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcaheList = nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcaheList))>1:
        for k in validEcaheList:
            if k==i:
                continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei-Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej =Ek
        return maxK,Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS,j)
    return j,Ej
def updateEk(oS,k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
'''
#完整Platt SMO算法中大优化例程
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i]*Ei)<-oS.tol)and (oS.alphas[i]<oS.c)) or\
        ((oS.labelMat[i]*Ei)>oS.tol) and (oS.alphas[i]>0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.c, oS.c+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0,oS.alphas[j]+oS.alphas[i]-oS.c)
            H = min(oS.c, oS.alphas[j]+oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0*oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - \
            oS.X[j,:]*oS.X[j,:].T
        if(eta >= 0):
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001)
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*\
            (alphaJold - oS.alphas[j])
        updateEk(oS,i)
        b1 = oS.b -Ei -oS.labelMat[i]*(oS.alphas[i]-alphaIold*\
            oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j])*\
            (oS.alphas[j] - alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej -oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
            oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*\
            (oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if(0<oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif(0<oS.alphas[j]) and (oS.c > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0
'''
#使用核函数需要对innerL及calcEk函数进行修改
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i]*Ei<-oS.tol)and (oS.alphas[i]<oS.C)) or \
        ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0*oS.K[i,j]-oS.K[i,i] - oS.K[j,j]
        if(eta >= 0):
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clibAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*\
            (alphaJold - oS.alphas[j])
        updateEk(oS,i)
        b1 = oS.b -Ei -oS.labelMat[i]*(oS.alphas[i]-alphaIold*\
            oS.K[i,i]-oS.labelMat[j])*\
            (oS.alphas[j] - alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej -oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
            oS.K[i,j]*-oS.labelMat[j]*\
            (oS.alphas[j]-alphaJold)*oS.K[j,j]
        if(0<oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif(0<oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0                

 #完整版platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            print ("fullSet, iter:%d i: %d,pairs changed %d "%\
                (iter,i,alphaPairsChanged))
            iter +=1
        else:
            nonBoundIs = nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound,iter:%d i:%d, pairs changed %d"%\
                    (iter, i, alphaPairsChanged))
            iter +=1
        if entireSet:
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

#计算ws
def calcWs(alphas, dataAttr, classLabels):
    X = mat(dataAttr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
#求出w之后，就可以利用X*W.T+b得出正负，为正是1类，为负是0类

#核转换函数
#kTup:存储核函数信息，kTup[0]核函数类型，其他的值为核函数可能需要的可选参数
def kernelTrans(X,A, kTup): 
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin' :
        K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] -A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("Houston We Have a problem--\
            That Kernel is not recognized")
    return K




def testRbf(k1 = 1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt') #读入数据
    b,alphas = smoP(dataArr, labelArr, 200,0.0001,10000,('rbf',k1)) #用smo算法得到b和alpha
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd] #获取alpha部位0的支持向量
    labelSV = labelMat[svInd] 
    print("there are %d Support Vectors" % shape(sVs)[0]) 
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:],('rbf',k1))  #利用结构化方法得到转换后的数据
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!= sign(labelArr[i]):
            errorCount +=1
    print("the training error rate is : %f " %(float(errorCount)/m))
    dataArr,labelArr = loadDataSet("testSetRBF2.txt")
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if(sign(predict)!= sign(labelArr[i])):
            errorCount += 1
    print("the test error rate is : %f" % (float(errorCount)/m))

#测试步骤：
#1，读入数据；
#2，用smo算法求得alpha和b；
#3，使用核函数转换；
#4，预测；
#5判断类别，同时可以统计错误率

#####
#下面是实现手写数字识别
#####
#图像信息以文本格式存储。图像已经处理成具有相同色彩和大小：宽高是32*32的黑白图像
#该函数是将32*32的图像信息转换成1*1024的向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' %(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr,200,0.0001,10000,kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d suppport vectors" %shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernalEval = kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernalEval.T*multiply(labelSV,alphas[svInd])+b
        if(sign(predict) != sign(labelArr[i])):
            errorCount +=1
    print("the training error rate is: %f" %(float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:],kTup)
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if(sign(predict) != sign(labelArr[i])):
            errorCount +=1
    print("the training error rate is: %f" %(float(errorCount)/m))