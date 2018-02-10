#Apriori算法中的辅助函数
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet): #创建单项产品的项集
    C1 = []
    for transaction in dataSet: #访问所有数据记录
        for item in transaction: #遍历记录中的每一条数据
            if not [item] in C1: #若数据不在C1列表中
                C1.append(([item])) #添加
    C1.sort()
    return map(frozenset,C1) #对C1中每个项集构建一个不变集合

def scanD(DMap,Ck,minSupport):  #返回一个包含支持度值的字典以备使用
    DList = list(DMap)
    CkList= list(Ck)
    ssCnt = {}
    numItems = float(len(DList)) #样本的数目
    for tid in DList: #遍历数据集
            for can in CkList: #遍历候选集合列表
                if can.issubset(tid): #项集是数据集的子集
                    #if not ssCnt.has_key(can): #若不是，添加上
                    if can not in ssCnt: #若不是，添加上
                        ssCnt[can] = 1
                    else: #若是，则数目加1
                        ssCnt[can] += 1
    retList = []
    supportData = {}
    for key in ssCnt: #遍历项集的数目记录
        support = ssCnt[key]/numItems #判断是否超过支持度如果超过，则保留
        if support >= minSupport: #retList中存储的是满足条件的项目集
            retList.insert(0,key) 
        supportData[key] = support #最频繁项集的支持度
    return retList,supportData

#Apriori算法
def aprioriGen(Lk,k): #输入参数为频繁项集列表Lk与项集元素个数k，输出为Ck，得到两两合并后的项集
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j]) #将两个集合合并
    return retList

def apriori(dataSet,minSupport=0.5):#apriori算法
    C1 = createC1(dataSet) #创建初始集合
    DMap = map(set,dataSet)
    L1,supportData = scanD(DMap,C1,minSupport) #删除掉不符合条件的集合
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k) #根据删除后的结合，两两合并，得到新的集合
        DMap = map(set,dataSet)
        Lk,supK = scanD(DMap,Ck,minSupport) #删除掉不符合条件的集合
        supportData.update(supK) #更新支持的集合列表
        L.append(Lk)
        k += 1
    return L,supportData


#关联规则生成函数
def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1,len(L)): #遍历频繁项集，只访问有两个或更多元素的集合
        for freqSet in L[i] : 
            H1 = [frozenset([item]) for item in freqSet]
            if(i > 1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData,br1,minConf=0.7): #评估规则，保留满足条件的规则
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if(conf>=minConf):
            print(freqSet-conseq,'-->',conseq,'conf',conf)
            br1.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7): #生成候选规则集
    m = len(H[0])
    if(len(freqSet)>(m+1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if(len(Hmp1)>1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


#下面是示例的代码，直接拷贝书中的程序,未验证
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print ("           -------->")
        for item in ruleTup[1]:
            print (itemMeaning[item])
        print ("confidence: %f" % ruleTup[2])
        print       #print a blank line
        
            
from time import sleep
'''
from votesmart import votesmart
votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print ('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print ("problem getting bill %d" % billNum)
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
'''        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print ('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print ("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning

