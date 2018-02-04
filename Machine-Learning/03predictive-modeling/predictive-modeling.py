# -*- coding: utf-8 -*-
"""
@Created on 2018/1/12 0012 上午 11:27

@author: ZhifengFang
"""
# 常用函数
'''
# 加载数据 
def load_data(input_file):
    X = []
    y = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])
    X = np.array(X)
    y = np.array(y)
    return X, y


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
'''

import numpy as np
import matplotlib.pyplot as plt

# 用SVM建立线性分类器
'''
# 使用第2章的创建简单分类器将数据分类并画出
# 1、加载数据

input_file = 'data_multivar.txt'

# input_file = 'data_multivar_imbalance.txt'

X, y = load_data(input_file)

# 2、分类
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

# 3、画图
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], facecolor='black', edgecolors='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolor='none', edgecolors='black', marker='s')
plt.show()

# 使用SVM
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
# params = {'kernel':'poly','degree':3}
params = {'kernel': 'rbf'}
# params = {'kernel': 'linear', 'class_weight': 'balanced'}
classifier = SVC(**params)
# 训练线性SVM分类器，并查看结果边界
classifier.fit(X_train, y_train)
plot_classifier(classifier, X_train, y_train)
# 测试数据集
y_test_pred = classifier.predict(X_test)
plot_classifier(classifier, X_test, y_test)
# 查看数据的精准性，训练数据集的分类报告
from sklearn.metrics import classification_report

print(
    classification_report(y_train, classifier.predict(X_train), target_names=['Class-' + str(int(i)) for i in set(y)]))
# 测试数据集的分类报告
print(classification_report(y_test, classifier.predict(X_test), target_names=['Class-' + str(int(i)) for i in set(y)]))
'''
# 提取置信度
'''
input_datapoints = np.array([[2, 1.5], [8, 9], [4.8, 5.2], [4, 4], [2.5, 7], [7.6, 2], [5.4, 5.9]])

for i in input_datapoints:
    print(i, '-->', classifier.decision_function(i)[0])  # 测量点到边界的距离

params = {'kernel': 'rbf', 'probability': True}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

for i in input_datapoints:
    print(i, '-->', classifier.predict_proba(i)[0])  # 这里要求params中probability必须为True，计算输入数据点的置信度

plot_classifier(classifier, input_datapoints, [0] * len(input_datapoints))
'''

# 寻找最优超参数
'''
# 1、加载数据，通过交叉验证
parameter_grid = [{'kernel': ['linear'], 'C': [1, 10, 50, 600]},
                  {'kernel': ['poly'], 'degree': [2, 3]},
                  {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 10, 50, 600]},
                  ]

metrics = ['precision', 'recall_weighted']

from sklearn import svm, grid_search, cross_validation
from sklearn.metrics import classification_report
# 2、为每个指标搜索最优超参数
for metric in metrics:
    classifier = grid_search.GridSearchCV(svm.SVC(C=1), parameter_grid, cv=5, scoring=metric)# 获取对象
    classifier.fit(X_train, y_train)# 训练
    for params, avg_score, _ in classifier.grid_scores_:# 看指标得分
        print(params, '-->', round(avg_score, 3))
    print('最好参数集：',classifier.best_params_)# 最优参数集
    y_true, y_pred = y_test, classifier.predict(X_test)
    print(classification_report(y_true, y_pred))# 打印一下性能报告
'''

# 建立时间预测器
'''
# 1、读取数据
input_file='building_event_multiclass.txt'
# input_file='building_event_binary.txt'

X=[]
y=[]
with open(input_file,'r') as f:
    for line in f.readlines():
        data=line[:-1].split(',')
        X.append([data[0]]+data[2:])
X=np.array(X)
# 2、编码器编码
from sklearn import preprocessing
label_encoder=[]
X_encoder=np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoder[:,i]=X[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoder[:,i]=label_encoder[-1].fit_transform(X[:,i])
X=np.array(X_encoder[:,:-1]).astype(int)
y=np.array(X_encoder[:,-1]).astype(int)
# 3、进行分类
from sklearn.svm import SVC
params={'kernel':'rbf','probability':True,'class_weight':'balanced'}
classifier=SVC(**params)
classifier.fit(X,y)
# 4、交叉验证
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(classifier,X,y,scoring='accuracy',cv=3)
print('accuracy:',accuracy.mean())
# 5、对新数据进行验证
input_data = ['Tuesday', '12:30:00','21','23']
input_data_encoder=[-1]*len(input_data)
count=0

for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoder[i]=int(input_data[i])
    else:
        label=[]
        label.append(input_data[i])
        input_data_encoder[i]=label_encoder[count].transform(label)
        count=count+1

result=int(classifier.predict(np.array(input_data_encoder)))
print('result:',label_encoder[-1].inverse_transform(result))
'''

# 估算交通流量

# 1、获取数据
X=[]
input_file='traffic_data.txt'
with open(input_file,'r') as f:
    for line in f.readlines():
        data=line[:-1].split(',')
        X.append(data)

X=np.array(X)

# 2、编码

from sklearn import preprocessing
label_encoder=[]
X_encoder=np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoder[:,i]=X[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoder[:,i]=label_encoder[-1].fit_transform(X[:,i])

X=X_encoder[:,:-1].astype(int)
y=X_encoder[:,-1].astype(int)
'''
# 3、线性回归
from sklearn.svm import SVR
# params = {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2}
params={'kernel':'rbf','C':10.0,'epsilon':0.2}# C表示对分类的惩罚，参数epsilon表示不使用惩罚的限制
regressor=SVR(**params)
regressor.fit(X,y)
with open('svr_eample.pkl','wb') as f:#由于训练太慢 所以可以将模型保存起来
    p.dump(regressor,f)
'''
import pickle  as p
with open('svr_eample.pkl','rb') as f:
    regressor=p.load(f)
# 4、验证
from sklearn.metrics import mean_absolute_error
y_pred=regressor.predict(X)
print('mean_absolute_error:',mean_absolute_error(y,y_pred))

# 5、预测新值


input_data = ['Tuesday', '13:35', 'San Francisco', 'yes']
input_data_encoder=[-1]*len(input_data)
count=0
for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoder[i]=int(input_data[i])
    else:
        label=[]
        label.append(input_data[i])
        input_data_encoder[i]=int(label_encoder[count].transform(label))
        count=count+1
        
result=regressor.predict(input_data_encoder)
print(result)


