# -*- coding: utf-8 -*-
"""
@Created on 2018/2/27 0027 下午 4:55

@author: ZhifengFang
"""

import pandas as pd

# 逻辑回归
'''

filename = 'bankloan.xls'
from sklearn.linear_model import RandomizedLogisticRegression as RLR# 随机逻辑回归模型
from sklearn.linear_model import LogisticRegression as LR# 逻辑回归模型

data = pd.read_excel(filename)
x = data.iloc[:, :8].as_matrix()
y = data.iloc[:, 8].as_matrix()

rlr = RLR()
rlr.fit(x, y)
print('score：', ','.join(data.columns[rlr.get_support()]))# 通过随机逻辑回归模型找出特征

x=data[data.columns[rlr.get_support()]].as_matrix()
lr=LR()
lr.fit(x,y)
print('正确率：',lr.score(x,y))# 获取训练结果
'''

# 决策树
'''
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

filename = 'sales_data.xls'
data = pd.read_excel(filename, index_col='序号')
data[data == '好'] = 1
data[data == '是'] = 1
data[data == '高'] = 1
data[data != 1] = -1

x = data.iloc[:, :3].as_matrix().astype(int)
y = data.iloc[:, 3].as_matrix().astype(int)

dtc = DTC(criterion='entropy')
dtc.fit(x, y)
x = pd.DataFrame(x)
with open('tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=x.columns, out_file=f)
'''

# 神经网络预测销量高低
'''
filename = 'sales_data.xls'
data = pd.read_excel(filename, index_col='序号')
data[data == '好'] = 1
data[data == '是'] = 1
data[data == '高'] = 1
data[data != 1] = -1

x = data.iloc[:, :3].as_matrix().astype(int)
y = data.iloc[:, 3].as_matrix().astype(int)

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()  # 建立模型
model.add(Dense(input_dim=3, units=10))
model.add(Activation('rel'))  # 用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(input_dim=10, units=1))
model.add(Activation('sigmoid'))  # 由于是0-1输出，用sigmoid函数作为激活函数

model.compile(loss='binary_crossentropy', optimizer='adam')
# 编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy，以及模式为binary
# 另外常见的损失函数还有mean_squared_error、categorical_crossentropy等，请阅读帮助文件。
# 求解方法我们指定用adam，还有sgd、rmsprop等可选

model.fit(x, y, epochs=1000, batch_size=10)  # 训练模型，学习一千次
yp = model.predict_classes(x).reshape(len(y))  # 分类预测

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

cm_plot(y, yp).show()
'''

# k-means聚类
'''
inputfile = 'consumption_data.xls'
outputfile = 'data-type.xls'

k = 3
iteration = 500
data = pd.read_excel(inputfile, index_col='Id')
data_zs = 1.0 * (data - data.mean()) / data.std()  # 标准化

from sklearn.cluster import KMeans

model = KMeans(n_clusters=k, init='k-means++', n_init=4)  # 分为k类
model.fit(data_zs)
# 简单打印结果
r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data.columns) + ['类别数目']  # 重命名表头
print(r)

# 详细输出原始数据及其类别
r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)  # 详细输出每个样本对应的类别
r.columns = list(data.columns) + ['聚类类别']  # 重命名表头
r.to_excel(outputfile)  # 保存结果


def density_plot(data):  # 自定义作图函数
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
    [p[i].set_ylabel('密度') for i in range(k)]
    plt.legend()
    return plt


# pic_output = 'pd_'  # 概率密度图文件名前缀
# for i in range(k):
#     density_plot(data[r['聚类类别'] == i]).savefig('%s%s.png' % (pic_output, i))

from sklearn.manifold import TSNE
# 聚类可视化工具
tsne = TSNE()
tsne.fit_transform(data_zs)  # 进行数据降维
tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)  # 转换数据格式

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 不同类别用不同颜色和样式绘图
d = tsne[r['聚类类别'] == 0]
plt.plot(d[0], d[1], 'r.')
d = tsne[r['聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
d = tsne[r['聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
plt.show()
'''

# Apriori算法
'''
# 自定义连接函数，用于实现L_{k-1}到C_k的连接
def connect_string(x, ms):
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r


# 寻找关联规则的函数
def find_rule(d, support, confidence, ms='--'):
    result = pd.DataFrame(index=['support', 'confidence'])  # 定义输出结果

    support_series = 1.0 * d.sum() / len(d)  # 支持度序列
    column = list(support_series[support_series > support].index)  # 初步根据支持度筛选
    k = 0

    while len(column) > 1:
        k = k + 1
        print('\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        print('数目：%s...' % len(column))
        sf = lambda i: d[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数

        # 创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(list(map(sf, column)), index=[ms.join(i) for i in column]).T

        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)  # 计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])  # 定义置信度序列

        for i in column2:  # 计算置信度序列
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i) - 1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]
    result = result.T.sort_values(['confidence', 'support'], ascending=False)  # 结果整理，输出
    print('\n结果为：')
    print(result)
    return result


inputfile = 'menu_orders.xls'
outputfile = 'apriori_rules.xls'

data = pd.read_excel(inputfile, header=None)
ct = lambda x: pd.Series(1, index=x[pd.notnull(x)])
b = map(ct, data.as_matrix())
data = pd.DataFrame(list(b)).fillna(0)
print('转换ok')
del b
support = 0.2
confidence = 0.5
ms = '---'
find_rule(data, support, confidence, ms).to_excel(outputfile)
'''

'''
# 参数初始化
discfile = 'arima_data.xls'
forecastnum = 5

# 读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
data = pd.read_excel(discfile, index_col='日期')

# 时序图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data.plot()
plt.show()

# 自相关图
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(data).show()

# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF

print('原始序列的ADF检验结果为：', ADF(data['销量']))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 差分后的结果
D_data = data.diff().dropna()
D_data.columns = ['销量差分']
D_data.plot()  # 时序图
plt.show()
plot_acf(D_data).show()  # 自相关图
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(D_data).show()  # 偏自相关图
print('差分序列的ADF检验结果为：', ADF(D_data['销量差分']))  # 平稳性检测

# 白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox

print('差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值

from statsmodels.tsa.arima_model import ARIMA

data['销量'] = data['销量'].astype(float)
# 定阶
pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
bic_matrix = []  # bic矩阵
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:  # 存在部分报错，所以用try来跳过报错。
            tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
print('BIC最小的p值和q值为：%s、%s' % (p, q))
model = ARIMA(data, (p, 1, q)).fit()  # 建立ARIMA(0, 1, 1)模型
model.summary2()  # 给出一份模型报告
model.forecast(5)  # 作为期5天的预测，返回预测结果、标准误差、置信区间。
'''

import numpy as np

# 参数初始化
inputfile = 'consumption_data.xls'  # 销量及其他属性数据
k = 3  # 聚类的类别
threshold = 2  # 离散点阈值
iteration = 500  # 聚类最大循环次数
data = pd.read_excel(inputfile, index_col='Id')  # 读取数据
data_zs = 1.0 * (data - data.mean()) / data.std()  # 数据标准化

from sklearn.cluster import KMeans

model = KMeans(n_clusters=k, init='k-means++', n_init=4)  # 分为k类，并发数4
model.fit(data_zs)  # 开始聚类

# 标准化数据及其类别
r = pd.concat([data_zs, pd.Series(model.labels_, index=data.index)], axis=1)  # 每个样本对应的类别
r.columns = list(data.columns) + ['聚类类别']  # 重命名表头

norm = []
for i in range(k):  # 逐一处理
    norm_tmp = r[['R', 'F', 'M']][r['聚类类别'] == i] - model.cluster_centers_[i]
    norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)  # 求出绝对距离
    norm.append(norm_tmp / norm_tmp.median())  # 求相对距离并添加

norm = pd.concat(norm)  # 合并

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
norm[norm <= threshold].plot(style='go')  # 正常点

discrete_points = norm[norm > threshold]  # 离群点
discrete_points.plot(style='ro')

for i in range(len(discrete_points)):  # 离群点做标记
    id = discrete_points.index[i]
    n = discrete_points.iloc[i]
    plt.annotate('(%s, %0.2f)' % (id, n), xy=(id, n), xytext=(id, n))

plt.xlabel('编号')
plt.ylabel('相对距离')
plt.show()
