# -*- coding: utf-8 -*-
"""
@Created on 2018/3/30 0030 下午 5:05

@author: ZhifengFang
"""

# 电力窃漏电用户自动识别
import pandas as pd
from random import shuffle

# 导入数据并打乱数据
inputfile = 'model.xls'
data = pd.read_excel(inputfile)
data = data.as_matrix()
shuffle(data)

# 将数据分为80%的训练和20%的测试
num_train = int(len(data) * 0.8)
train = data[:num_train, :]
test = data[num_train:, :]

def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数

    cm = confusion_matrix(y, yp)  # 混淆矩阵

    import matplotlib.pyplot as plt  # 导入作图库
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar()  # 颜色标签

    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    return plt


# 使用LM神经网络
'''
from keras.models import Sequential  # 导入神经网络初始化函数
from keras.layers.core import Dense, Activation  # 导入神经网路层函数和激活函数

netfile = 'net.model'
net = Sequential()  # 建立神经网络
net.add(Dense(input_dim=3, units=10))
net.add(Activation('relu'))
net.add(Dense(input_dim=10, units=1))
net.add(Activation('sigmoid'))
net.compile(loss='binary_crossentropy', optimizer='adam')  # 编译模型
net.fit(train[:, :3], train[:, 3], epochs=1000, batch_size=1)  # 训练模型1000次
net.save_weights(netfile)  # 保存模型

predict_result = net.predict_classes(train[:, :3]).reshape(len(train))  # 预测结果变形

cm_plot(train[:, 3], predict_result).show()
'''
# 使用CART决策树
from sklearn.tree import DecisionTreeClassifier

treefile = 'tree.pkl'
tree = DecisionTreeClassifier()
tree.fit(train[:, :3], train[:, 3])

from sklearn.externals import joblib
joblib.dump(tree, treefile)  # 保存模型

from sklearn.metrics import confusion_matrix #导入混淆矩阵函数
cm_plot(train[:, 3], tree.predict(train[:, :3])).show()

cm = confusion_matrix(train[:,3], tree.predict(train[:,:3])) #混淆矩阵

import matplotlib.pyplot as plt #导入作图库

from sklearn.metrics import roc_curve #导入ROC曲线函数
fpr, tpr, thresholds = roc_curve(test[:,3], tree.predict_proba(test[:,:3])[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果