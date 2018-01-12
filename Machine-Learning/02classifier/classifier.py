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
# 逻辑回归分类器
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

from sklearn.ensemble import RandomForestClassifier
# 根据汽车特征评估质量
'''
from sklearn import preprocessing

# 准备数据
input_file = 'car.data.txt'

X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')  # line[:-1]表示line中最后一个换行删除
        X.append(data)

X = np.array(X)

# 使用标记编将字符串转化为数值
label_encoder = []
X_encoder = np.empty(X.shape)

for i, item in enumerate(X[0]):  # 由于相同的信息是以列的形式显示，所以应该按列进行标记编码
    label_encoder.append(preprocessing.LabelEncoder())  # 初始化每列的标记编码器
    X_encoder[:, i] = label_encoder[-1].fit_transform(X[:, i])  # 未标记编码

X = X_encoder[:, :-1].astype(int)  # 将所有数据的除最后一列作为X，最后一列作为y
y = X_encoder[:, -1].astype(int)

# 训练随机森林分类器
params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}  # 跟上章监督学习中的随机森林回归的参数一个意思：
# n_estimators指评估器的数量，则决策树数量，min_samples_split指决策树分裂一个节点需要用到的最小数据样本量
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)
'''
# 进行交叉验证
'''
from sklearn import model_selection

# model_selection 将之前的sklearn.cross_validation, sklearn.grid_search 和 sklearn.learning_curve模块组合到一起

accuracy = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print('accuracy:', str(round(accuracy.mean(), 2)) + '%')

# 对某条信息进行分类
input_data = ['low', 'low', '4', 'more', 'big', 'med']
input_data_encoded = [-1] * len(input_data)

for i, item in enumerate(input_data):
    labels=[]
    labels.append(input_data[i])# 转换形式，否则下行会报错
    input_data_encoded[i] = int(label_encoder[i].transform(labels))

input_data_encoder = np.array(input_data_encoded)
output_class = classifier.predict(input_data_encoder)  # 预测
print('结果：', label_encoder[-1].inverse_transform(output_class)[0])  # 最后一个编码器是结果
'''

# 验证曲线
'''
from sklearn.model_selection import  validation_curve

classifier=RandomForestClassifier(max_depth=4,random_state=7)
parameter_grid=np.linspace(25,200,8).astype(int)
train_scores,validation_scores=validation_curve(classifier,X,y,'n_estimators',parameter_grid,cv=5)#对n_estimators参数进行验证
print('training scores:',train_scores)
print('validation scores:',validation_scores)

plt.figure()
plt.plot(parameter_grid,100*np.average(train_scores,axis=1),color='black')
plt.show()

classifier=RandomForestClassifier(n_estimators=20,random_state=7)
parameter_grid=np.linspace(2,10,5).astype(int)
train_scores,validation_scores=validation_curve(classifier,X,y,'max_depth',parameter_grid,cv=5)#max_depth
print('training scores:',train_scores)
print('validation scores:',validation_scores)

plt.figure()
plt.plot(parameter_grid,100*np.average(train_scores,axis=1),color='black')
plt.show()

'''

# 学习曲线
'''
from sklearn.model_selection import learning_curve

classifier=RandomForestClassifier(random_state=7)
parameter_grid=np.array([200,500,800,1100])
train_size,train_scores,validation_scores=learning_curve(classifier,X,y,train_sizes=parameter_grid,cv=5)#cv表示五折交叉验证
print('train_scores:',train_scores)
print('validation_scores:',validation_scores)

plt.figure()
plt.plot(parameter_grid,100*np.average(train_scores,axis=1),color='black')
plt.show()

'''

# 估算收入阶层

# 1、读取数据
input_file='adult.data.txt'
X=[]

countLess=0
countMore=0
countAll=20000

with open(input_file,'r') as f:
    for line in f.readlines():
        if '?' not in line:
            data=line[:-1].split(', ')
            # 2、若大部分点都属于同一个类型，则分类器会倾向于该类型，所以应该选出大于50k与小于等于50k各10000
            if data[-1]=='<=50K' and countLess<countAll:
                X.append(data)
                countLess=countLess+1
            elif data[-1]=='>50K' and countMore<countAll:
                X.append(data)
                countMore=countMore+1
            if countMore>=countAll and countLess>=countAll:
                break;

X=np.array(X)
from sklearn import preprocessing
# 3、对数据进行编码
label_encoder=[]
for i,item in enumerate(X[0]):
    if item.isdigit():
        X[:,i]=X[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X[:,i]=label_encoder[-1].fit_transform(X[:,i])

y=X[:,-1].astype(int)
X=X[:,:-1].astype(int)
# 4、将数据分成训练和测试

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=5)
# 5、训练数据
classifier_gaussiannb=GaussianNB()
classifier_gaussiannb.fit(X_train,y_train)
y_test_pred=classifier_gaussiannb.predict(X_test)
# 6、提取性能指标
f1=cross_val_score(classifier_gaussiannb,X,y,scoring='f1_weighted',cv=5)
print('f1:',str(round(f1.mean()*100,2))+'%')
# 7、预测新的值
input_data = ['39', 'State-gov', '77516', 'Bachelors', '13', 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male', '2174', '0', '40', 'United-States']
count=0
input_data_encoder=[-1]*len(input_data)
for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoder[i]=int(input_data[i])
    else:
        labels = []
        labels.append(input_data[i])
        input_data_encoder[i]=int(label_encoder[count].transform(labels))
        count=count+1

result=classifier_gaussiannb.predict(input_data_encoder)
result=label_encoder[-1].inverse_transform(result)
print('resutl:',result)








