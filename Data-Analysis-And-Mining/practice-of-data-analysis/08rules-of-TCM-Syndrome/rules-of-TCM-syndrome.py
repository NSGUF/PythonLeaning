# -*- coding: utf-8 -*-
"""
@Created on 2018/4/2 0002 下午 2:47

@author: ZhifengFang
"""
import pandas as pd
from sklearn.cluster import KMeans

'''
processedfile = 'data_processed.xls'  # 数据处理后文件
data = pd.read_excel('data.xls')
typelabel = {'肝气郁结证型系数': 'A', '热毒蕴结证型系数': 'B', '冲任失调证型系数': 'C', '气血两虚证型系数': 'D', '脾胃虚弱证型系数': 'E', '肝肾阴虚证型系数': 'F'}
keys = list(typelabel.keys())
result = pd.DataFrame()
if __name__ == '__main__':  # 判断是否主窗口运行，如果是将代码保存为.py后运行，则需要这句，如果直接复制到命令窗口运行，则不需要这句。
    for i in range(len(keys)):
        # 调用k-means算法，进行聚类离散化
        print('正在进行“%s”的聚类...' % keys[i])
        kmodel = KMeans(n_clusters=4)  # n_jobs是并行数，一般等于CPU数较好
        kmodel.fit(data[[keys[i]]].as_matrix())  # 训练模型

        r1 = pd.DataFrame(kmodel.cluster_centers_, columns=[typelabel[keys[i]]])  # 聚类中心
        r2 = pd.Series(kmodel.labels_).value_counts()  # 分类统计
        r2 = pd.DataFrame(r2, columns=[typelabel[keys[i]] + 'n'])  # 转为DataFrame，记录各个类别的数目
        r = pd.concat([r1, r2], axis=1).sort_values(typelabel[keys[i]])  # 匹配聚类中心和类别数目
        r.index = [1, 2, 3, 4]

        r[typelabel[keys[i]]] = pd.rolling_mean(r[typelabel[keys[i]]], 2)  # rolling_mean()用来计算相邻2列的均值，以此作为边界点。
        r[typelabel[keys[i]]][1] = 0.0  # 这两句代码将原来的聚类中心改为边界点。
        result = result.append(r.T)

    result = result.sort_index(axis=0, ascending=True)  # 以Index排序，即以A,B,C,D,E,F顺序排# 以Index排序，即以A,B,C,D,E,F顺序排
    result.to_excel(processedfile)
'''
import time #导入时间库用来计算用时

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

    result = result.T.sort_values(by=['confidence', 'support'], ascending=False)  # 结果整理，输出
    print('\n结果为：')
    print(result)

    return result


inputfile = 'apriori.txt' #输入事务集文件
data = pd.read_csv(inputfile, header=None, dtype = object)

start = time.clock() #计时开始
print('\n转换原始数据至0-1矩阵...')
ct = lambda x : pd.Series(1, index = x[pd.notnull(x)]) #转换0-1矩阵的过渡函数
b = map(ct, data.as_matrix()) #用map方式执行
data = pd.DataFrame(list(b)).fillna(0) #实现矩阵转换，空值用0填充
end = time.clock() #计时结束
print('\n转换完毕，用时：%0.2f秒' %(end-start))
del b #删除中间变量b，节省内存

support = 0.06 #最小支持度
confidence = 0.75 #最小置信度
ms = '---' #连接符，默认'--'，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符

start = time.clock() #计时开始
print('\n开始搜索关联规则...')
find_rule(data, support, confidence, ms)
end = time.clock() #计时结束
print('\n搜索完成，用时：%0.2f秒' %(end-start))