# -*- coding: utf-8 -*-
"""
@Created on 2018/4/2 0002 上午 9:39

@author: ZhifengFang
"""

import pandas as pd

datafile = 'air_data.csv'
resultfile = 'explore.xls'

# data = pd.read_csv(datafile, encoding='utf-8')
# print(type(data))
# 对数据进行缺失值和异常值分析
'''
explore = data.describe(percentiles=[], include='all').T  # percentiles是指计算多少的分位数表，

explore['null'] = len(data) - explore['count']  # 计算获得为空的个数
explore = explore[['null', 'max', 'min']]
explore.columns = ['空数值', '最大值', '最小值']
explore.to_excel(resultfile)
'''

# 数据清洗
cleanfile = 'data_clean.xls'
'''
data = data[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()]  # 票价非空才保留
# 只保留票价非零或平均折扣率与总飞行公里数同时为0
index1 = data['SUM_YR_1'] != 0
index2 = data['SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0) #该规则是“与”
data = data[index1 | index2 | index3]
data.to_excel(cleanfile)
'''
# 属性规约
'''
data = pd.read_excel(cleanfile)
filename='zscore1data.xls'
data=data[['LOAD_TIME','FFP_DATE','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
data.to_excel(filename,index=False)

# 数据变换

filename = 'zscore1data.xls'
data = pd.read_excel(filename)

datazs=pd.DataFrame(columns=['L','R','F','M','C'])
datazs['L']=data['avg_discount']
datazs['R']=data['LAST_TO_END']
datazs['F']=data['FLIGHT_COUNT']
datazs['M']=data['SEG_KM_SUM']
datazs['C']=data['avg_discount']
datazs.sort_values(by='C')
datazs.to_excel('zscore2data.xls',index=False)
'''
# 标准差标准化
'''
dataza = pd.read_excel('zscoredata.xls')
zscoredfile='zscoreddate1.xls'
dataza=(dataza-dataza.mean(axis=0))/(dataza.std(axis=0))
dataza.columns=['Z'+i for i in dataza.columns]
dataza.to_excel(zscoredfile,index=False)
'''

# K-Means聚类算法
from sklearn.cluster import KMeans

data = pd.read_excel('zscoreddata.xls')
kmodel=KMeans(n_clusters=5,init='k-means++',n_init=4)
kmodel.fit(data)
print(kmodel.cluster_centers_)
print(kmodel.labels_)
