# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:55:02 2017

@author: ZhifengFang

官方文档：http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
"""
#机器学习算法无法理解原始数据，所以需对原始数据进行预处理，常用预处理如下：
import numpy as np
from sklearn import preprocessing
# data=np.array([
#         [3,-1.5,2,-5.4],
#         [0,4,-0.3,2.1],
#         [1,3.3,-1.9,-4.3]
#         ])
# #1.均值移除（Mean removal) 将特征数据的分布调整成标准正太分布，也叫高斯分布，
# #也就是使得数据的均值维0，方差为1.标准化是针对每一列而言的
# #方法一
# data_standardized=preprocessing.scale(data)
# print('Mean=',data_standardized.mean(axis=0))#特征均值几乎为0
# print('Std=',data_standardized.std(axis=0))#标准差为1
# #方法二
# scaler=preprocessing.StandardScaler().fit(data)
# print('Mean=',scaler.transform(data).mean(axis=0))#特征均值几乎为0
# print('Std=',scaler.transform(data).std(axis=0))#标准差为1
#
# #2.范围缩放（Scaling)：为了对付那些标准差相当小的特征并且保留下稀疏数据中的0值
# #方法一：计算公式如下：
# #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# #X_scaled = X_std / (max - min) + min
# data_minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# data_scaled = data_minmax_scaler.fit_transform(data)
# print("Min max scaled data:", data_scaled)
# #方法二：特征中绝对值最大的那个数为1，其他数以此维标准分布在[[-1，1]之间
# max_abs_scaler = preprocessing.MaxAbsScaler()
# x_train_maxsbs = max_abs_scaler.fit_transform(data)
# print("Max abs scaled data:", x_train_maxsbs)
#
# # 3.归一化（Normalization）->正则化：保证每个特征向量的值都缩放到相同的数值范围内，
# #提高不同特征特征数据的可比性，如数据有许多异常值可使用此方法
# #方法一：第二个参数可谓l1与l2，最常用为调整到l1范数，使所有特征向量之和为1
# data_normalized = preprocessing.normalize(data, norm='l1')
# print("\nL1 normalized data:\n", data_normalized)
# #方法二：
# normalizer = preprocessing.Normalizer(copy=True, norm='l2').fit(data)#创建正则器
# normalizer.transform(data)
#
# # 二值化（Binarization）将数值型的特征数据转换成布尔类型的值
# # 方法一
# data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)#比1.4大的为1，小的为0
# print("\nBinarized data:\n", data_binarized)
# #方法二：
# binarizer = preprocessing.Binarizer(threshold=0)
# print("\nBinarized data:\n", binarizer.transform(data))
#
#
# # 独热编码（One-Hot Encoding）http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# encoder = preprocessing.OneHotEncoder()
# encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
# encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
# print("Encoded vector:", encoded_vector)
# print("_values_:", encoder.n_values_)#值的每个特征的最大数量。
# print("Encoded vector:", encoder.feature_indices_)
#
#
# # 弥补缺失值
# imp = preprocessing.Imputer(missing_values='NaN', axis=0)#NaN可换成其他
# imp.fit([[1, 2], [np.nan, 3], [7, 6]])
# x = [[np.nan, 2], [6, np.nan], [7, 6]]
# print(imp.transform(x))#填入(1+7)/2和(2+3+6)/3
#
# # 生成多项式的特征，得到高阶相互作用特征
# poly = preprocessing.PolynomialFeatures(2)# 创建2次方的多项式
# print(poly.fit_transform(data))
#
# #定制变压器:辅助数据清洗或处理
# transformer = preprocessing.FunctionTransformer(np.log1p)
# print(transformer.transform(np.array([[0, 1], [2, 3]])))

#==============================================================================
# 参考文献：http://blog.csdn.net/sinat_33761963/article/details/53433799
# http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
# Python机器学习经典实例
# 
#==============================================================================


# 标记编码的方法 把单词标记转换成数值形式，让算法懂得如何操作标记。
# # 1、定义一个标记编码器（label encoder）
# label_encoder=preprocessing.LabelEncoder()
# # 2、给出标记
# input_classes=['abc','egf','opq','rst','bcd','cde']
# # 3、为标记编码，按第一个字母顺序为标记排编号，从0开始
# label_encoder.fit(input_classes)
# for i,item in enumerate(label_encoder.classes_):
#     print(item,'->',i)
# # 4、转换标记，将给出的标记变为排序标号
# label=['rst','opq','abc','rst','opq','abc']
# encoded_labels=label_encoder.transform(label)
# print(label,encoded_labels)
# # 5、反转标记，将给出的数字，对应出标记
# encoded_label=[4,3,2,4,1]
# labels=label_encoder.inverse_transform(encoded_label)
# print(encoded_label,labels)


# import numpy as np
# # 创建线性回归器 目的：提取输入变量与输出变量的关联线性模型，可以使得实际输出与线性方程预
# # 测的输出的差平方和（sum of squares of differences）最小化，该方法称普通最小二乘法（Ordinary Least Squares，OLS）
# # 1、获取数据并解析数据到变量X和Y中
# X=[]
# Y=[]
# with open('data_singlevar.txt','r') as f:
#     for line in f.readlines():
#         data = [float(i) for i in line.split(',')]
#         xt,yt=data[:-1],data[-1]
#         X.append(xt)
#         Y.append(yt)
# # 2、将数据分为训练集和测试集，各为80%，20%
# num_training=int(0.8*len(X))
# num_test=len(X)-num_training
#
# X_training=np.array(X[0:num_training])
# Y_training=np.array(Y[0:num_training])
#
# X_test=np.array(X[num_training:])
# Y_test=np.array(Y[num_training:])
#
# # 3、创建回归器对象
# from sklearn import linear_model
# linear_regressor=linear_model.LinearRegression()
#
# linear_regressor.fit(X_training,Y_training)#训练
#
# # 4、获取预测数据，并将其显示
# import matplotlib.pyplot as plot
#
# y_train_pred=linear_regressor.predict(X_training)
#
# plot.figure()
# plot.scatter(X_training,Y_training,color='green')
# plot.plot(X_training,y_train_pred,color='black')
# plot.title('train数据显示')
# plot.show()
#
#
# # 5、验证测试数据，并显示
#
# y_test_pred=linear_regressor.predict(X_test)
# plot.figure()
# plot.scatter(X_test,Y_test,color='green')
# plot.plot(X_test,y_test_pred,color='black')
# plot.title('test数据显示')
# plot.show()
#
# # 评价回归器的拟合效果，用误差表示实际值与模型预测值之间的差值
# # 简述几个衡量回归器效果的重要指标（metric）：
# import sklearn.metrics as sm
#
# print('平均绝对误差：',round(sm.mean_absolute_error(Y_test,y_test_pred)))
# print('均方误差：',round(sm.mean_squared_error(Y_test,y_test_pred)))
# print('中位数绝对误差：',round(sm.median_absolute_error(Y_test,y_test_pred)))
# print('解释方差分：',round(sm.explained_variance_score(Y_test,y_test_pred)))
# print('R方得分：',round(sm.r2_score(Y_test,y_test_pred)))
#
# # # 保存模型
# # import pickle  as p
# # with open('saved_model_output.pkl','wb') as f:
# #     p.dump(linear_regressor,f)
# #
# # import pickle as p
# # #加载模型
# # with open('saved_model_output.pkl','rb') as f:
# #     model_liner=p.load(f)
# # y_pre=model_liner.predict(X_test)
# #
# #
# # import matplotlib.pyplot as plot
# # y_test_pred=model_liner.predict(X_test)
# # plot.figure()
# # plot.scatter(X_test,Y_test,color='green')
# # plot.plot(X_test,y_test_pred,color='black')
# # plot.title('test数据显示')
# # plot.show()
#
#
# #岭回归
#
# #alpha趋于0时，岭回归器就是普通最小乘法的线性回归器，若希望对异常值不敏感，设为大一点
# ridge_regressor=linear_model.Ridge(alpha=1,fit_intercept=True,max_iter=10000)
# ridge_regressor.fit(X_training,Y_training)
# y_test_pred_ridge=ridge_regressor.predict(X_test)
# plot.figure()
# plot.scatter(X_test,Y_test,color='green')
# plot.plot(X_test,y_test_pred_ridge,color='black')
# plot.title('train数据显示')
# plot.show()
# print('平均绝对误差：',round(sm.mean_absolute_error(Y_test,y_test_pred_ridge)))
# print('均方误差：',round(sm.mean_squared_error(Y_test,y_test_pred_ridge)))
# print('中位数绝对误差：',round(sm.median_absolute_error(Y_test,y_test_pred_ridge)))
# print('解释方差分：',round(sm.explained_variance_score(Y_test,y_test_pred_ridge)))
# print('R方得分：',round(sm.r2_score(Y_test,y_test_pred_ridge)))
#
# # 创建多项式回归器
# quadratic_featurizer = preprocessing.PolynomialFeatures(degree=5)#获取多项式对象
# X_train_quadratic = quadratic_featurizer.fit_transform(X_training)#获得多项式形式的输入
# xx=np.linspace(-6,4,100)#曲线显示
# regressor_quadratic = linear_model.LinearRegression()
# regressor_quadratic.fit(X_train_quadratic, Y_training)
# xx_quadratic = quadratic_featurizer.fit_transform(xx.reshape(xx.shape[0], 1))#获得多项式形式的输入
# yy_pre=regressor_quadratic.predict(xx_quadratic)#获取预测值
#
# plot.figure()
# plot.scatter(X_training,Y_training,color='green')
# plot.plot(xx,yy_pre , 'r-')
# plot.title('train数据显示')
# plot.show()

# 估算房屋价格
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error,explained_variance_score
import matplotlib.pyplot as plt

# 1、获取标准房屋价格数据库，scikit-learn提供接口
housing_data=datasets.load_boston()
# 2、将数据分入到X，Y中,并通过shuffle打乱数据，random_state控制如何打乱顺序
X,y=shuffle(housing_data.data,housing_data.target,random_state=7)
# 3、80%训练数据，20%测试数据
num_training=int(0.8*len(X))
X_train,y_train=X[0:num_training],y[0:num_training]
X_test,y_test=X[num_training:],y[num_training:]
# 4、拟合决策树模型，并限制最大深度为4
dt_regressor=DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train,y_train)
# 5、用带AdaBoost算法
ab_regressor=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=4)
ab_regressor.fit(X_train,y_train)
# 6、评估决策树模型测试结果，尽量保证均方误差最低，而解释方差分最高
y_dt_pred=dt_regressor.predict(X_test)
print('均方差：',mean_squared_error(y_test,y_dt_pred))
print('解释方差：',explained_variance_score(y_test,y_dt_pred))
# 7、评估AdaBoost测试结果，同上
y_ab_pred=ab_regressor.predict(X_test)
print('均方差：',mean_squared_error(y_test,y_ab_pred))
print('解释方差：',explained_variance_score(y_test,y_ab_pred))


def plot_feature_importances(feature_importances,title,feature_names):
    feature_importances=100*(feature_importances/max(feature_importances))
    index_sorted=np.flipud(np.argsort(feature_importances))#argsort获得数值从小到大排序的索引，flipud反序
    pos=np.arange(index_sorted.shape[0])+0.5

    plt.figure()
    plt.bar(pos,feature_importances[index_sorted],align='center')
    plt.xticks(pos,feature_names[index_sorted])
    plt.title(title)
    plt.show()

plot_feature_importances(dt_regressor.feature_importances_,'dt',housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_,'dt',housing_data.feature_names)















