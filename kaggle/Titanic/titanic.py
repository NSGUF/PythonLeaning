# -*- coding: utf-8 -*-
"""
@Created on 2018/5/31 13:15

@author: ZhifengFang
"""

import numpy as np
import pandas as pd

inputfile = 'train.csv'


def yuchuli(inputfile, outputfile):
    data = pd.read_csv(inputfile, delimiter=',', header=0)
    datanew = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    datanew["Age"] = datanew["Age"].fillna(datanew["Age"].median())

    datanew.loc[datanew["Sex"] == "male", "Sex"] = 0
    datanew.loc[datanew["Sex"] == "female", "Sex"] = 1
    datanew.loc[datanew["Embarked"] == "S", "Embarked"] = 0
    datanew.loc[datanew["Embarked"] == "C", "Embarked"] = 1
    datanew.loc[datanew["Embarked"] == "Q", "Embarked"] = 2
    datanew["Embarked"] = datanew["Embarked"].fillna(datanew["Embarked"].median())

    datanew["Fare"] = (datanew["Fare"] - min(datanew["Fare"])) / (max(datanew["Fare"]) - min(datanew["Fare"]))

    datanew.to_csv(outputfile, index=False)


# yuchuli(inputfile)
# yuchuli('test.csv','testnew.csv')
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


def classifyTest():
    train = pd.read_csv('train_new.csv', header=0)
    # test=pd.read_csv('test_new.csv',header=0)
    num_test = int(0.2 * train.shape[0])
    X = np.array(train)
    y = X[:, 0]
    X = X[:, 1:]
    errorcount = 0
    for i in range(num_test):
        label = classify0(X[i, :], X[num_test:, :], y[num_test:], 3)
        if label != y[i]:
            errorcount += 1
    print('正确率为：', 1 - errorcount / float(num_test))


classifyTest()


def classifyOutPut():
    train = pd.read_csv('train_new.csv', header=0)
    test = pd.read_csv('testnew.csv', header=0)
    X_test = np.array(test)
    X_test = X_test[:, 1:]
    print(X_test)
    X = np.array(train)
    y_train = X[:, 0]
    X_train = X[:, 1:]
    print(X_test.shape)
    y_pre = []
    errorcount = 0
    for i in range(len(X_test)):
        label = classify0(X_test[i, :], X_train, y_train, 3)
        y_pre.append(int(label))
    result = pd.read_csv('gender_submission.csv')
    result['Survived'] = y_pre
    result.to_csv('gender_submission.csv', index=False)


# classifyOutPut()
def randomForest():
    from sklearn.ensemble import RandomForestRegressor
    train = pd.read_csv('train_new.csv', header=0)
    # test=pd.read_csv('test_new.csv',header=0)
    num_training = int(0.2 * train.shape[0])
    X = np.array(train)
    y = X[:, 0]
    X = X[:, 1:]
    X_test = np.array(X[0:num_training])
    Y_test = np.array(y[0:num_training])

    X_training = np.array(X[num_training:])
    Y_training = np.array(y[num_training:])

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_training, Y_training)
    y_pre = rf_regressor.predict(X_test)
    errorcount = 0
    for i in range(len(y_pre)):
        if y_pre[i] != Y_test[i]:
            errorcount += 1

    print('错误率：', errorcount / float(len(y_pre)), errorcount)
    # print(y_pre)
