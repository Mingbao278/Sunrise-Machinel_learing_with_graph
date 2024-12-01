import operator
from math import log

# 创建数据集
def createDataSet():  
    dataSet =  [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
                ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
                ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
                ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
                ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
                ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
                ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
                ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
                ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
                ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
                ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
                ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']]
    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']  
    return dataSet, features

#计算数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    #{是：11, 否：6
    #}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0 # 如果当前标签没有在字典中，添加进去
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 按某个特征分类后的数据
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最优特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy

        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i 

    return bestFeature

#选取分类后样本数量最多的类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#构建决策树
def createTree(dataSet,features):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):#标签都相同
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = features[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(features[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subFeatures = features[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subFeatures)
    return myTree

if __name__ == '__main__':
    dataSet, features = createDataSet()
    print(createTree(dataSet, features))