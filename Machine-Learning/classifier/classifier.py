# -*- coding: utf-8 -*-
"""
@Created on 2017/12/13 21:43

@author: ZhifengFang
"""
"""
# 简单分类器
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
X = np.array([[3,1], [2,5], [1,8], [6,4], [5,2], [3,5], [4,7], [4,-1]])
y = [0, 1, 1, 0, 0, 1, 1, 0]
# 根据y的值分类X，取值范围为0~N-1，N表示有N个类
class_0=np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1=np.array([X[i] for i in range(len(X)) if y[i]==1])
# 将点画出
plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],color='red',marker='s')
plt.scatter(class_1[:,0],class_1[:,1],color='black',marker='x')
# 创建y=x的直线
line_x=range(10)
line_y=line_x
plt.plot(line_x,line_y,color='blue',linewidth=3)
plt.show()
"""
#逻辑回归分类器
'''


import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# 准备数据
X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# 初始化一个逻辑分类回归器
classifier=linear_model.LogisticRegression(solver='liblinear',C=10000)#solver设置求解系统方程的算法类型，C表示正则化强度，越小表强度越高,C越大，各个类型的边界更优。

#训练分类器
classifier.fit(X,y)

# 画出数据点和边界
plot_classifier(classifier,X,y)
'''
import matplotlib.pyplot as plt
import numpy as np

# 定义画图函数
def plot_classifier(classifier, X, y):
    # 获取x，y的最大最小值，并设置余值
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0] + 1.0)
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1] + 1.0)
    # 设置网格步长
    step_size = 0.01
    # 设置网格
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    # 计算出分类器的分类结果
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)
    # 画图
    plt.figure()
    # 选择配色方案
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    # 画点
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)
    # 设置图片取值范围
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    # 设置x与y轴
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
    plt.show()

# 朴素贝叶斯分类器
'''
from sklearn.naive_bayes import GaussianNB

# 准备数据
input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)
# 建立朴素贝叶斯分类器
classifier_gaussiannb=GaussianNB()
classifier_gaussiannb.fit(X,y)
y_pre=classifier_gaussiannb.predict(X)
# 计算分类器的准确性
accuracy=100.0*(y==y_pre).sum()/X.shape[0]
print('结果:',accuracy)
# 画出数据和边界
plot_classifier(classifier_gaussiannb,X,y)
'''

# 分割训练集和测试集
'''
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

# 准备数据
input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)
x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.25,random_state=5)# 测试数据占25%，
# 建立朴素贝叶斯分类器
classifier_gaussiannb=GaussianNB()
classifier_gaussiannb.fit(x_train,y_train)
y_test_pre=classifier_gaussiannb.predict(x_test)
# 计算分类器的准确性
accuracy=100.0*(y_test==y_test_pre).sum()/x_test.shape[0]
print('结果:',accuracy)
# 画出数据和边界
plot_classifier(classifier_gaussiannb,x_test,y_test_pre)
'''

# 性能指标
'''
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

# 准备数据
input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)
x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.25,random_state=5)# 测试数据占25%，
# 建立朴素贝叶斯分类器
classifier_gaussiannb=GaussianNB()
classifier_gaussiannb.fit(x_train,y_train)
y_test_pre=classifier_gaussiannb.predict(x_test)
# 计算分类器的准确性
# accuracy=100.0*(y_test==y_test_pre).sum()/x_test.shape[0]

num_validations = 5
# 正确率
accuracy = cross_validation.cross_val_score(classifier_gaussiannb,X, y, scoring='accuracy', cv=num_validations)
print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")
# F1
f1 = cross_validation.cross_val_score(classifier_gaussiannb,X, y, scoring='f1_weighted', cv=num_validations)
print("F1: " + str(round(100*f1.mean(), 2)) + "%")
# 精度
precision = cross_validation.cross_val_score(classifier_gaussiannb,X, y, scoring='precision_weighted', cv=num_validations)
print("Precision: " + str(round(100*precision.mean(), 2)) + "%")
# 召回率
recall = cross_validation.cross_val_score(classifier_gaussiannb,X, y, scoring='recall_weighted', cv=num_validations)
print("Recall: " + str(round(100*recall.mean(), 2)) + "%")
# 画出数据和边界
plot_classifier(classifier_gaussiannb,x_test,y_test_pre)

'''

# 混淆矩阵
'''
from sklearn.metrics import confusion_matrix

# 显示混淆矩阵
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.gray)
    plt.colorbar()
    tick_marks=np.arange(4)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.show()

y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
confusion_mat=confusion_matrix(y_true,y_pred)
plot_confusion_matrix(confusion_mat)
'''

# 提取性能报告
'''
from sklearn.metrics import classification_report

target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
print(classification_report(y_true,y_pred,target_names=target_names))
'''

# 根据汽车特征评估质量

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# 准备数据
input_file='car.data.txt'

X=[]
count=0
with open(input_file,'r') as f:
    for line in f.readlines():
        data=line[:-1].split(',')
        X.append(data)

X=np.array(X)

# 使用标记编将字符串转化为数值































