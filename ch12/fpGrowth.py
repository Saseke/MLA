######################################
##这一章的程序，分析了一遍，但没太看懂，做个记录
######################################

class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue #节点名字
        self.count = numOccur #计数
        self.nodeLink = None #连接相似的元素项
        self.parent = parentNode #指向父节点
        self.children = {} #存放节点的子节点
    def inc(self,numOccur):
        self.count += numOccur
    def disp(self,ind=1): #将树以文本形式显示
        print(' '*ind,self.name,' ',self.count) 
        for child in self.children.values():
            child.disp(ind+1)


#FP树构建函数
def createTree(dataSet, minSup=1):
    headerTable = {} #头指针
    for trans in dataSet:
        for item in trans :
            headerTable[item] = headerTable.get(item,0)+dataSet[trans] #更新头指针对应数目增加
    for k in list(headerTable.keys()): #遍历头指针，是否满足支持度条件，不满足则删除
        if headerTable[k]<minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys()) #元素项去重
    if len(freqItemSet) == 0: #如果没有元素项满足要求，就退出
        return None,None
    for k in headerTable: #
        headerTable[k] = [headerTable[k],None] #改变头指针的结构，原先只记录了数目，现在加了一项指向指针
    retTree = treeNode('Null Set',1,None)
    for tranSet,count in dataSet.items(): #项集以及其对应出现的数目
        localD = {}
        for item in tranSet: #项集里的元素
            if item in freqItemSet: #如果在头指针里有该元素，满足支持度条件
                localD[item] = headerTable[item][0] #获取该元素出现的数目
        if len(localD) > 0: #元素不为空
            orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)] #对列表排序
            updateTree(orderedItems,retTree,headerTable,count) #排序后，更新FP树
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):
    if items[0] in inTree.children: #如果该元素中已含有该元素
        inTree.children[items[0]].inc(count) #元素数目加1
    else: #若不包含
        inTree.children[items[0]] = treeNode(items[0],count,inTree) #创建一个节点
        if headerTable[items[0]][1] == None: #如果头指针表中，该元素的指向为NONE，就赋值为指向该节点
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]]) #更新node的指向
    if len(items)>1: #
        updateTree(items[1::],inTree.children[items[0]],headerTable,count) #更新items[0]下对应的子树


def updateHeader(nodeToTest,targetNode):
    while(nodeToTest.nodeLink != None): #循环判断nodeToTest的linkNode是否为空，直到找到为None为止，然后再指向targetNodt
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r','z','h','j','p'],
                ['z','y','x','w','v','u','t','s'],
                ['z'],
                ['r','x','n','o','s'],
                ['y','r','x','z','q','t','p'],
                ['y','z','x','e','q','s','t','m']]
    return simpDat
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1 #创建字典，以项集为索引，数目设置为1
    return retDict

#从FP中挖掘频繁项集
#条件模式基是以所查找元素项为结尾的路径集合
#发现以给定元素项结尾的所有路径的函数
def ascendTree(leafNode, prefixPath): #根据叶节点，迭代上溯整棵树，找路径
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
def findPrefixPath(basePat,treeNode):
    condPats = {}
    while treeNode != None: #遍历链表
        prefixPath = []
        ascendTree(treeNode, prefixPath) #根据节点，回溯找到到达该节点的路径
        if len(prefixPath)>1: #如果路径上不只一个元素
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats #条件模式基字典

#递归查找频繁项集的mineTree函数
def mineTree(inTree,headerTable,minSup, preFix, freqItemList):
    bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:p[1])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree,myHead = createTree(condPattBases,minSup)
        if myHead != None:
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)