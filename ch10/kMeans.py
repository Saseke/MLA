#K-均值聚类支持函数
from numpy import *

def loadDataSet(fileName): #读文件
    dataMat = []
    fr = open(fileName) 
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB): #欧氏距离,此外还有马氏距离，明氏距离等
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k): #该函数为给定数据集构建一个包含k个随机质心的集合。
    n = shape(dataSet)[1] #属性个数
    centroids = mat(zeros((k,n))) #每个属性都对应k个簇
    for j in range(n): #遍历属性
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ) #保证随机质心必须在整个数据集的边界之内
        centroids[:,j] = minJ+rangeJ*random.rand(k,1) #对每个属性都找到k个质心 
    return centroids #返回k*n个质心

#K均值聚类算法
def kMeans(dataSet,k,distMeans=distEclud,createCent=randCent):
    m = shape(dataSet)[0] #获取样本数目
    clusterAssment = mat(zeros((m,2))) 
    #clusterAssment用于存储簇分配结果矩阵，第一列记录簇的索引值，第二列存储误差。
    #这里存储误差是指当前点到簇质心的距离，对应了所有的样本
    centroids = createCent(dataSet,k) #初始化质心，这里是进行的随机选择
    clusterChanged = True #是否继续迭代标志
    while clusterChanged:
        clusterChanged = False
        for i in range(m): #遍历样本点
            minDist = inf #最小距离
            minIndex = -1 #最小距离对应的质心索引
            for j in range(k): #遍历所有簇的质心
                distJI = distMeans(centroids[j,:],dataSet[i,:]) #计算当前点到质心的距离 
                if (distJI < minDist): #选择一个距离最近的质心
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex: #如果当前样本点得到的最近的簇的索引不等于最小距离对应的索引
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 #
        print (centroids) #打印簇质心
        for cent in range(k): #遍历质心
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]] #获取数据集中归属于该簇的样本
            centroids[cent,:] = mean(ptsInClust,axis=0)#求均值，更新质心信息
    return centroids, clusterAssment #返回质心和簇信息

#二分K均值聚类算法
def biKmeans(dataSet,k,distMeans=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0] #初始化一个簇
    print(type(centList))
    for j in range(m):
        clusterAssment[j,1] = distMeans(mat(centroid0),dataSet[j,:])**2 #计算每个点到质心的误差值
    while(len(centList)<k): #判断是否达到了用户要求的质心数
        lowestSSE = inf #最小偏差
        for i in range(len(centList)): #访问所有的质心
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] #获取隶属于当前簇的所有样本点
            centroidMat,splitClusterAss = kMeans(ptsInCurrCluster,2,distMeans) #k均值聚类，设置k=2
            sseSplit = sum(splitClusterAss[:,1]) #求得分割之后的所有误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #获取并不隶属于该簇的误差和
            print("sseSplit, and not split:",sseSplit,sseNotSplit)
            if(sseSplit+sseNotSplit)<lowestSSE: #如果分割之后的SSE小于最小的SSE
                bestCentToSplit = i #标记将第几个质心分割
                bestNewCents = centroidMat #分割之后的
                bestClustAss = splitClusterAss.copy() #分割之后的clusterAss
                lowestSSE = sseSplit+sseNotSplit #最小的SSE设置为分割之后的SSE
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(centList) #
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
        print("the bestCentToSplit is : ", bestCentToSplit)
        print("the len of bestClustAss is: ",len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].A[0].tolist() #将被分割的质点用新质点覆盖
        centList.append(bestNewCents[1,:].A[0].tolist()) #新的另一个质点添加到质点列表
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment


#####下面的程序，不涉及聚类原理性的东西，直接拷贝的是原程序文件。
import urllib
import json
def geoGrab(stAddress, city): #函数运行不过，提示连接失败
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params) #添加了 .parse，否则报出没有urlencode
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    #c=urllib.urlopen(yahooApi)   #此处会报出没有urlopen，python2和python3的区别
    c=urllib.request.urlopen(yahooApi) 
    #return json.loads(c.read())  #此处会报出JSON object must be str,not bytes
    return json.loads(c.read().decode("utf-8"))

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print ("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines  #经纬度转换为距离
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):  #簇绘图，numClust是希望的簇数目
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeans=distSLC)
    print(type(myCentroids))
    mat(myCentroids)
    print(myCentroids[:,0])
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

