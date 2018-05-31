# -*- coding: utf-8 -*-
"""
@Created on 2018/5/28 11:16

@author: ZhifengFang
"""
import numpy as np
import operator
import os


def createDateSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-近邻算法
def classify0(inX, dataSet, labels, k):  # inX：被预测的新数据点；dataSet：数据集；labels：对应的标签；
    dataSetCol = dataSet.shape[0]  # 获取数据集的行数
    inXMat = np.tile(inX, (dataSetCol, 1)) - dataSet  # 将输入值复制成数据集的行数并减去数据集，这样可得到被预测数据点与数据集中每个数据的差
    inXMat = inXMat ** 2  # 平方
    inXMat = inXMat.sum(axis=1)  # 按行求和
    inXMat = inXMat ** 0.5  # 开根号
    sortedIndic = inXMat.argsort()  # 排序的索引
    classCount = {}
    for i in range(k):
        klabels = labels[sortedIndic[i]]  # 前k个数的标签
        classCount[klabels] = classCount.get(klabels, 0) + 1  # 计算每个标签的个数
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)  # 按照字典的第一个数排序
    return sortedClassCount[0][0]
# dataSet, labels = createDateSet()
# print(classify0([0, 0], dataSet, labels, 3))
# 读取文件并将内容分别存入数组和列表中
def filetomatrix(filename):
    f = open(filename)
    lines = f.readlines()
    matcol = len(lines)
    mat = np.empty(shape=(matcol, 3))
    classLabels = []
    for i, line in zip(range(matcol), lines):
        line = line.strip().split('\t')  # 去除空格，并按照\t分割
        mat[i, :] = line[:3]
        classLabels.append(int(line[-1]))
    return mat, classLabels
# mat, classLabels = filetomatrix('datingTestSet2.txt')
# 归一化处理，公式为：(x-min_x)/(max_x-min_x)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.empty(shape=np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
# 测试，将0.9的数据为训练数据集，将0.1的数据设置测试数据，并打印正确率
def datingClassTest():
    dataMat, dataLabels = filetomatrix('datingTestSet2.txt')
    dataMat, ranges, minVals =autoNorm(dataMat)
    m = len(dataLabels)
    num_test = int(0.5 * m)
    num_error = 0
    for i in range(num_test):
        label = classify0(dataMat[i, :], dataMat[num_test:m, :], dataLabels[num_test:m], 3)
        if label != dataLabels[i]:
            num_error += 1
    print('错误率为：', num_error / float(num_test))
    print(num_error)
# datingClassTest()
# 输入一个人的数据，并预测
def classifyPersion():
    result=['不喜欢','一般','喜欢']
    x1=float(input('x1'))
    x2=float(input('x2'))
    x3=float(input('x3'))
    test=[x1,x2,x3]
    mat,labels=filetomatrix('datingTestSet2.txt')
    mat,ranges,min=autoNorm(mat)
    resultlabel=classify0(test,mat,labels,3)
    print(result[resultlabel-1])
# classifyPersion()
def imgtovector(filename):
    resultVec=[]
    f=open(filename)
    lines=f.readlines()
    for line in lines:
        for a in line.strip():
            resultVec.append(int(a))
    return resultVec
# print(len(imgtovector('C:\\Users\\ZhifengFang\\Desktop\\machinelearninginaction\\Ch02\\testDigits\\0_13.txt')))
# 获得手写数据集的数据集和标签集，输入：文件夹地址
def getHandWritingData(dirPath):
    filelist=os.listdir(dirPath)# 获取文件夹中的文件名列表
    m=len(filelist)
    dataMat,dataLabels=[],[]
    for filename in filelist:# 循环文件
        dataMat.append(imgtovector(dirPath+'\\'+filename))
        dataLabels.append(int(filename.split('_')[0]))
    return np.array(dataMat),dataLabels
#
def handWritingClassTest():
    testMat,testLabels=getHandWritingData('C:\\Users\\ZhifengFang\\Desktop\\machinelearninginaction\\Ch02\\testDigits')
    trainMat,trainLabels=getHandWritingData('C:\\Users\\ZhifengFang\\Desktop\\machinelearninginaction\\Ch02\\trainingDigits')
    num_error=0
    for i in range(len(testLabels)):
        label=classify0(testMat[i,:],trainMat,trainLabels,3)
        if label!=testLabels[i]:
            num_error+=1
    print('错误率：',num_error/float(len(testLabels)))

handWritingClassTest()


