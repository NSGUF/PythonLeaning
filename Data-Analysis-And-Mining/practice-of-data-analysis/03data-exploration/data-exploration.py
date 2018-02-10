# -*- coding: utf-8 -*-
"""
@Created on 2018/2/6 0006 下午 1:43

@author: ZhifengFang
"""
# 异常值检查
'''
import pandas as pd
import matplotlib.pyplot as plt
catering_sale='catering_sale.xls' # 餐饮数据
data=pd.read_excel(catering_sale,index_col='日期')# 读取数据并指定列为日期
print(data.describe())# 打印出数据的基本数据，count表示非空值数
print(len(data))# 总共条数

plt.figure()
p=data.boxplot(return_type='dict')# 画箱型图
x=p['fliers'][0].get_xdata()
y=p['fliers'][0].get_ydata()
y.sort()
for i in range(len(x)):
    if i>0:
        plt.annotate(y[i],xy=(x[i],y[i]),xytext=(x[i]+0.05-0.8/(y[i]-y[i-1]),y[i]))
    else:
        plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.8, y[i]))
plt.show()
'''

# 餐饮销量数据统计量分析
'''
import pandas as pd
catering_scale='catering_sale.xls'
data=pd.read_excel(catering_scale,index_col='日期')
data=data[(data['销量']>400)&(data['销量']<5000)]# 过滤异常值

statistics=data.describe()# 获取结果

statistics.loc['range']=statistics.loc['max']-statistics.loc['min']# 添加极值
statistics.loc['var']=statistics.loc['std']-statistics.loc['mean']# 添加异变系数
statistics.loc['dis']=statistics.loc['75%']-statistics.loc['25%']# 四分位间距
print(statistics)
'''

# 帕累托图
'''
import pandas as pd
import matplotlib.pyplot as plt

catering_dish = 'catering_dish_profit.xls'
data = pd.read_excel(catering_dish, index_col='菜品名')  # 读取信息
print(data)
data = data['盈利'].copy()

data = data.sort_values(ascending=False)  # 对盈利倒序

plt.figure()
data.plot(kind='bar')
p = 1.0 * data.cumsum() / data.sum()# 比例
p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
plt.annotate(format(p[6], '.4%'), xy=(6, p[6]), xytext=(6 * 0.9, p[6] * 0.9),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))  # 添加注释，即85%处的标记。这里包括了指定箭头样式。
plt.show()
'''

# 不同菜品之间的关系
import pandas as pd

data=pd.read_excel('catering_sale_all.xls',index_col='日期')
print(data.corr())# 相关系数矩阵，即给出了任意两个菜之间的关系 Spearman(Pearman)
print(data.corr(method='spearman'))# Spearman
print(data.corr()['百合酱蒸凤爪']) # 给出了百合酱蒸凤爪与其他任意菜之间的关系