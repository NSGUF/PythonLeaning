# -*- coding: utf-8 -*-
"""
@Created on 2018/4/9 22:26

@author: ZhifengFang
"""

# 属性变换
import pandas as pd

'''
#参数初始化
discfile = 'discdata.xls' #磁盘原始数据
transformeddata = 'discdata_processed.xls' #变换后的数据

data = pd.read_excel(discfile)
data = data[data['TARGET_ID'] == 184].copy() #只保留TARGET_ID为184的数据

data_group = data.groupby('COLLECTTIME') #以时间分组

def attr_trans(x): #定义属性变换函数
  result = pd.Series(index = ['SYS_NAME', 'CWXT_DB:184:C:\\', 'CWXT_DB:184:D:\\',	'COLLECTTIME'])
  result['SYS_NAME'] = x['SYS_NAME'].iloc[0]
  result['COLLECTTIME'] = x['COLLECTTIME'].iloc[0]
  result['CWXT_DB:184:C:\\'] = x['VALUE'].iloc[0]
  result['CWXT_DB:184:D:\\'] = x['VALUE'].iloc[1]
  return result

data_processed = data_group.apply(attr_trans) #逐组处理
data_processed.to_excel(transformeddata, index = False)
'''
'''
# 参数初始化
discfile = 'discdata_processed.xls'
predictnum = 5  # 不使用最后5个数据

data = pd.read_excel(discfile)
data = data.iloc[: len(data) - 5]  # 不检测最后5个数据

# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF

diff = 0
adf = ADF(data['CWXT_DB:184:D:\\'])
while adf[1] > 0.05:
    diff = diff + 1
    adf = ADF(data['CWXT_DB:184:D:\\'].diff(diff).dropna())

print(u'原始序列经过%s阶差分后归于平稳，p值为%s' % (diff, adf[1]))
'''

'''
#参数初始化
discfile = 'discdata_processed.xls'

data = pd.read_excel(discfile)
data = data.iloc[: len(data)-5] #不使用最后5个数据

#白噪声检测
from statsmodels.stats.diagnostic import acorr_ljungbox

[[lb], [p]] = acorr_ljungbox(data['CWXT_DB:184:D:\\'], lags = 1)
if p < 0.05:
  print(u'原始序列为非白噪声序列，对应的p值为：%s' %p)
else:
  print(u'原始该序列为白噪声序列，对应的p值为：%s' %p)

[[lb], [p]] = acorr_ljungbox(data['CWXT_DB:184:D:\\'].diff().dropna(), lags = 1)
if p < 0.05:
  print(u'一阶差分序列为非白噪声序列，对应的p值为：%s' %p)
else:
  print(u'一阶差分该序列为白噪声序列，对应的p值为：%s' %p)
  
'''

'''
#参数初始化
discfile = 'discdata_processed.xls'

data = pd.read_excel(discfile, index_col = 'COLLECTTIME')
data = data.iloc[: len(data)-5] #不使用最后5个数据
xdata = data['CWXT_DB:184:D:\\']

from statsmodels.tsa.arima_model import ARIMA

#定阶
pmax = int(len(xdata)/10) #一般阶数不超过length/10
qmax = int(len(xdata)/10) #一般阶数不超过length/10
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
    try: #存在部分报错，所以用try来跳过报错。
      tmp.append(ARIMA(xdata, (p,1,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值

p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(u'BIC最小的p值和q值为：%s、%s' %(p,q))
'''

'''
#参数初始化
discfile = 'discdata_processed.xls'
lagnum = 12 #残差延迟个数

data = pd.read_excel(discfile, index_col = 'COLLECTTIME')
data = data.iloc[: len(data)-5] #不使用最后5个数据
xdata = data['CWXT_DB:184:D:\\']

from statsmodels.tsa.arima_model import ARIMA #建立ARIMA(0,1,1)模型

arima = ARIMA(xdata, (0, 1, 1)).fit() #建立并训练模型
xdata_pred = arima.predict(typ = 'levels') #预测
pred_error = (xdata_pred - xdata).dropna() #计算残差

from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验

lb, p= acorr_ljungbox(pred_error, lags = lagnum)
h = (p < 0.05).sum() #p值小于0.05，认为是非白噪声。
if h > 0:
  print(u'模型ARIMA(0,1,1)不符合白噪声检验')
else:
  print(u'模型ARIMA(0,1,1)符合白噪声检验')
'''


#参数初始化
file = '../data/predictdata.xls'
data = pd.read_excel(file)

#计算误差
abs_ = (data[u'预测值'] - data[u'实际值']).abs()
mae_ = abs_.mean() # mae
rmse_ = ((abs_**2).mean())**0.5 # rmse
mape_ = (abs_/data[u'实际值']).mean() # mape

print(u'平均绝对误差为：%0.4f，\n均方根误差为：%0.4f，\n平均绝对百分误差为：%0.6f。' %(mae_, rmse_, mape_))