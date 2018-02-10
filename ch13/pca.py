
from numpy import *
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)
#返回的是降维后的原数据集和重构后的数据集
def pca(dataMat, topNfeat=9999999): 
    meanValues = mean(dataMat, axis = 0) #平均值
    meanRemoved = dataMat - meanValues #减去平均值
    covMat = cov(meanRemoved,rowvar = 0) #计算协方差
    eigValues, eigVectors = linalg.eig(mat(covMat)) #求得特征值和特征向量
    eigValInd = argsort(eigValues) #将特征值按从小到大排列
    eigValInd = eigValInd[:-(topNfeat+1):-1] #然后取最大的topNfeat个特征值
    regEigVectors = eigVectors[:,eigValInd] #eigValInd对应的特征向量
    lowDDataMat = meanRemoved*regEigVectors #将原始数据转换到新的空间
    reconMat = (lowDDataMat*regEigVectors.T)+meanValues #重构数据集
    return lowDDataMat,reconMat #

#利用PCA对半导体制造数据降维示例

#将NaN替换成平均值的函数
def replaceNanWithMean():
    dataMat = loadDataSet('secom.data',' ') #加载数据
    numFeat = shape(dataMat)[1] #特征值的个数
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i]) #求均值时把值为0的过滤掉了？？？
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal #替换
    return dataMat