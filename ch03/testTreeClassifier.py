import trees
import treePlotter

fr = open('lenses.txt') #打开数据文件
#读取文件记录，每行记录按tab键隔开
lenses = [record.strip().split('\t') for record in fr.readlines()] 
lensesLabels = ['age','prescript','astigmatic', 'tearRate'] #类别标签
lensesTree=trees.createTree(lenses,lensesLabels) #创建构造树
print(lensesTree)
treePlotter.createPlot(lensesTree) #通过matplotlib绘制决策树
