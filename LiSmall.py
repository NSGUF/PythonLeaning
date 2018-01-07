# -*- coding: utf-8 -*-
"""
@Created on 2017/12/23 23:56

@author: ZhifengFang
"""
import xlrd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score


# liData=xlrd.open_workbook('C:\\Users\\ZhifengFang\\Desktop\\LiSmall.xlsx')


def plot_feature_importances(feature_importances, title, feature_names):  # 画出所要测试与哪个属性关系最大
    feature_importances = 100 * (feature_importances / max(feature_importances))
    index_sorted = np.flipud(np.argsort(feature_importances))  # argsort获得数值从小到大排序的索引，flipud反序
    pos = np.arange(index_sorted.shape[0]) + 0.5

    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.title(title)
    plt.show()


# 评估共享单车重要因素

# 1、读取数据，该数据是从文件中读取
def load_datasets(filename):  # C:\\Users\\ZhifengFang\\Desktop\\LiSmall.xlsx
    liData = xlrd.open_workbook(filename)  # 获取工作簿
    tabel = liData.sheets()[0]  # 获取表

    X = []
    y = []
    nrows = tabel.nrows  # 获取总行数
    for i in range(nrows):
        X.append(tabel.row_values(i)[2:12])
        y.append(tabel.row_values(i)[1])

    feature_name = np.array(X[1])

    return np.array(X[2:]).astype(np.float32), np.array(y[2:]).astype(np.float32), feature_name


# 2、获取数据并将文件打乱放入X,y中
X, y, feature_name = load_datasets('C:\\Users\\ZhifengFang\\Desktop\\LiSmall.xlsx')
X, y = shuffle(X, y, random_state=7)
# X,y=preprocessing.scale(X),preprocessing.scale(y)
# 3、将数据分成0.8训练和0.2的测试数据
num_training = int(len(X) * 0.9)
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 4、训练回归
# #n_estimators指评估器的数量，则决策树数量，min_samples_split指决策树分裂一个节点需要用到的最小数据样本量
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
rf_regressor.fit(X_train, y_train)
x_point = X_test[4]
y_point = y_test[4]
# 4、拟合决策树模型，并限制最大深度为4
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)
# 5、用带AdaBoost算法
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=4)
ab_regressor.fit(X_train, y_train)

# 5、评价随机森林回归的效果
y_pre = rf_regressor.predict(X_test)
# 6、评估决策树模型测试结果，尽量保证均方误差最低，而解释方差分最高
y_dt_pred = dt_regressor.predict(X_test)

# 7、评估AdaBoost测试结果，同上
y_ab_pred = ab_regressor.predict(X_test)

print('rf结果：', rf_regressor.predict(x_point), '实际结果：', y_point)  # 测试结果
print('dt结果：', dt_regressor.predict(x_point), '实际结果：', y_point)  # 测试结果
print('ab结果：', ab_regressor.predict(x_point), '实际结果：', y_point)  # 测试结果

print('rf均方差：', mean_squared_error(y_test, y_pre))
print('rf解释方差：', explained_variance_score(y_test, y_pre))
print('rfR方', r2_score(y_test, y_pre))

print('dt均方差：', mean_squared_error(y_test, y_dt_pred))
print('dt解释方差：', explained_variance_score(y_test, y_dt_pred))
print('dtR方', r2_score(y_test, y_dt_pred))

print('ab均方差：', mean_squared_error(y_test, y_ab_pred))
print('ab解释方差：', explained_variance_score(y_test, y_ab_pred))
print('abR方', r2_score(y_test, y_ab_pred))

plot_feature_importances(dt_regressor.feature_importances_, 'dt', feature_name)
plot_feature_importances(ab_regressor.feature_importances_, 'ab', feature_name)
plot_feature_importances(rf_regressor.feature_importances_, 'rf', feature_name)








