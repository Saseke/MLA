def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#相似度的计算
from numpy import *
from numpy import linalg as la

def ecludSim(inA,inB): #欧氏距离相似度，越接近与1，相似度越高
    return 1.0/(1.0+la.norm(inA-inB))

def pearsSim(inA,inB): #皮尔逊相关系数计算
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1] #数值归一到在0到1之间

def cosSim(inA,inB): 
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom) #数值归一到0到1之间

#基于物品相似度的推荐引擎
def standEst(dataMat,user,simMeas,item): #数据矩阵、用户编号、相似度计算方法、物品
    n = shape(dataMat)[1] #特征数，在这里指物品数
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]  #用户是否对该物品做了评价
        if userRating == 0: #若没有评价，则跳过
            continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0] #寻找两个用户都评级的物品
        if len(overLap) == 0: #如果没有，相似度为0
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j]) #若有，计算相似度
        print("the %d and %d similarity is:%f" % (item,j,similarity))
        simTotal += similarity #相似度之和
        ratSimTotal += similarity*userRating #相似度与用户评分的乘积
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal #评分归一化
#推荐算法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1] #对给定用户，建立一个未评分的物品列表
    if(len(unratedItems)==0): #
        return 'you rated everything'
    itemScores = []
    for item in unratedItems: #遍历列表
        estimatedScore = estMethod(dataMat,user, simMeas, item) #预测相似度
        itemScores.append((item,estimatedScore)) 
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N] #对相似度进行排序，取相似度最高的N个物品

#基于SVD的评分估计
def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1] #获取物品数目
    simTotal = 0.0
    ratSimTotal = 0.0
    U,sigma,VT = la.svd(dataMat) #svd分解
    sig4 = mat(eye(4)*sigma[:4]) #将sigma转换成4*4的对角矩阵，其余的丢弃
    xformedItems = dataMat.T*U[:,:4]*sig4.I #将物品转换到低维空间
    for j in range(n):
        userRating = dataMat[user,j] #获取特定用户的对物品的评价
        if userRating == 0 or j == item: #如果评分为0或者等于原物品，跳过
            continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T) #在低维空间计算相似度
        print("the %d and %d similarity is: %f" %(item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

#基于SVD的图像压缩
def printMat(inMat, thresh=0.8):
    for i in range(32): #32*32像素的
        for k in range(32):
            if float(inMat[i,k])>thresh: #大于阈值，就打印1，否则打印0
                print (1,end = '')
            else:
                print (0,end = '')
        print ('')
def imgCompress(numSV=3, thresh=0.8):
    my1 = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i])) 
        my1.append(newRow) 
    myMat = mat(my1) #将文件数据存储到myMat中，32*31
    print ("*******original matrix")
    printMat(myMat,thresh) 
    U,sigma,VT = la.svd(myMat) #svd分解
    sigRecon = mat(zeros((numSV,numSV)))
    for k in range(numSV):
        sigRecon[k,k] = sigma[k]
    reconMat = U[:,:numSV]*sigRecon*VT[:numSV,:] #重构数据
    print("***reconstructed matrix using %d singular values****" % numSV)
    printMat(reconMat,thresh)