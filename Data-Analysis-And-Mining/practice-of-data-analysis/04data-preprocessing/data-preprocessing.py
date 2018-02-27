# -*- coding: utf-8 -*-
"""
@Created on 2018/2/10 0010 上午 10:10

@author: ZhifengFang
"""

# 拉格朗日插补
'''
import pandas as pd
from scipy.interpolate import lagrange

inputfile = 'catering_sale.xls'
outputfile = 'sales.xls'

data = pd.read_excel(inputfile)  # 读取excel
data.loc[(data['销量'] < 400) | (data['销量'] > 5000), '销量'] = None  # 异常值变为空值


def ployinterp_column(s, n, k=5):  # 默认是前后5个
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数，n的前后5个，这里有可能取到不存在的下标，为空
    y = y[y.notnull()]  # 如果y里面有空值的话就去掉
    return lagrange(y.index, list(y))(n)  # 最后的括号就是我们要插值的n


for i in data.columns:
    if i == '日期':
        continue
    for j in range(len(data)):
        if (data[i].isnull())[j]:  # 空值进行插值
            data.loc[j, i] = ployinterp_column(data[i], j)
data.to_excel(outputfile)
'''

# 数据规范化
'''
import pandas as pd
import numpy as np
datafile='normalization_data.xls'
data=pd.read_excel(datafile,header=None)
# 最小-最大规范化
print((data-data.min())/(data.max()-data.min()))
# 零-均值规范化
print((data-data.mean())/data.std())
# 小数定标规范化
print(data/10**np.ceil(np.log10(data.abs().max())))
'''

# 数据离散化
'''
import pandas as pd

datafile = 'discretization_data.xls'
data = pd.read_excel(datafile)
data = data['肝气郁结证型系数'].copy()
k=4
d1=pd.cut(data,k,labels=range(k))# 等宽离散法

# 等频率离散化
w = [1.0 * i / k for i in range(k + 1)]
w = data.describe(percentiles=w)[4:4 + k + 1]  # 使用describe函数自动计算分位数
w[0] = w[0] * (1 - 1e-10)
d2 = pd.cut(data, w, labels=range(k))

from sklearn.cluster import KMeans  # 引入KMeans

kmodel = KMeans(n_clusters=k, n_jobs=4)  # 建立模型，n_jobs是并行数，一般等于CPU数较好
kmodel.fit(data.reshape((len(data), 1)))  # 训练模型
c = pd.DataFrame(kmodel.cluster_centers_).sort(0)  # 输出聚类中心，并且排序（默认是随机序的）
w = pd.rolling_mean(c, 2).iloc[1:]  # 相邻两项求中点，作为边界点
w = [0] + list(w[0]) + [data.max()]  # 把首末边界点加上
d3 = pd.cut(data, w, labels=range(k))


def cluster_plot(d, k):  # 自定义作图函数来显示聚类结果
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure(figsize=(8, 3))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')

    plt.ylim(-0.5, k - 0.5)
    return plt


cluster_plot(d1, k).show()

cluster_plot(d2, k).show()
cluster_plot(d3, k).show()
'''

# 线损率属性构造
'''
import pandas as pd
inputfile='electricity_data.xls'
outfile='electricity_data_out.xls'
data=pd.read_excel(inputfile)
data['线损率']=(data['供入电量']-data['供出电量'])/data['供入电量']
data.to_excel(outfile,index=False)# index表示行号是否显示
'''

# 小波变换特征提取
'''
inputfile='leleccum.mat'

from scipy.io import loadmat
mat=loadmat(inputfile)# mat位python专属格式，需要用loadmat读取
signal=mat['leleccum'][0]
import pywt
coeffs=pywt.wavedec(signal,'bior3.7',level=5)# 小波变换特征提取，返回level+1个数字，第一个数组为逼近系数数组，后面的依次为细节系数数组
print(coeffs)
'''

# 主成分分析
'''
import pandas as pd
from sklearn.decomposition import PCA

inputfile = 'principal_component.xls'
outputfile = 'dimention_reducted.xls'
data = pd.read_excel(inputfile, header=None)

pca = PCA(n_components=None, copy=True,
          whiten=False)  # n_components表示算法中保留的主成分个数，int或string，即保留下来的特征个数，缺省时默认为None，所有成分被保留，为string时，如n_components=‘mle'表示自动选择特征个数n，使得满足所有要求的方差百分比，copy，默认为True，表示是否在运行算法时，将原始训练数据复制一份，若为True，则运行PCA算法后，原始数据不会有任何改变，因为是在副本上运行，若为False，则运行PCA算法后，原始数据会改变，即在原始数据中进行降维。whiten表是否白化，使得每个特征具有相同的方差。

pca.fit(data)
print(pca.components_)  # 返回模型的各个特征向量
print(pca.explained_variance_ratio_)  # 返回各个成分各自的方差百分比

pca = PCA(3)
pca.fit(data)
low_d = pca.transform(data)  # 降低维度
pd.DataFrame(low_d).to_excel(outputfile)  # 保存结果
pca.inverse_transform(low_d)  # 复原数据
'''
