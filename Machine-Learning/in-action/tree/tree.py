# -*- coding: utf-8 -*-
"""
@Created on 2018/5/31 15:37

@author: ZhifengFang
"""
import numpy as np


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


from math import log


# 计算该数据集的香农熵
def calcShannonEnt(dataSet):
    labelCounts = {}
    for data in dataSet:  # 计算数据集中每个标签的总个数
        label = data[-1]
        labelCounts[label] = labelCounts.get(label, 0) + 1
    shannonEnt = 0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / len(dataSet)  # 公式
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


dataSet, labels = createDataSet()


# print(calcShannonEnt(dataSet))
# 划分数据集，剔除axis列中和value不相同的数据（行），并删除axis列的值
def splitDataSet(dataSet, axis, value):
    resultSet = []
    for data in dataSet:
        if data[axis] == value:
            newData = data[:axis] + data[axis + 1:]
            resultSet.append(newData)
    return resultSet


# 选择最好的划分数据特征
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):  # 循环特征，为每个特征计算熵差，熵差最大的为
        listFeature = [example[i] for example in dataSet]
        features = set(listFeature)
        newEntropy = 0
        for feature in features:
            newDataSet = splitDataSet(dataSet, i, feature)
            newEntropy += len(newDataSet) / float(len(dataSet)) * calcShannonEnt(newDataSet)
        newEntropy = baseEntropy - newEntropy
        if newEntropy > bestInfoGain:
            bestInfoGain = newEntropy
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]


# 构造决策树
def createTree(dataSet, labels):
    classList = [example[0] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeture = chooseBestFeatureToSplit(dataSet)
    bestLabel = labels[bestFeture]
    myTree = {bestLabel: {}}
    del (labels[bestFeture])
    featValues = [example[bestFeture] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestLabel][value] = createTree(splitDataSet(dataSet, bestFeture, value), subLabels)
    return myTree


myTree = createTree(dataSet, labels)
print(myTree.keys())


# 决策树分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel
          

print(classify(myTree, ['no surfacing', 'flippers'], [1, 0]))


# 保存模型
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


# 加载模型
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


