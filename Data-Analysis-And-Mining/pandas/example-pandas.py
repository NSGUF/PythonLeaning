# -*- coding: utf-8 -*-
"""
@Created on 2017/12/13 21:43

@author: ZhifengFang
"""
import pandas as pd

# print(pd.__version__)

# Series对象
# s=pd.Series([1,3,2,4,5],index=['b','a','c','d','e'])#若index不指定，将自动创建位置下标的索引
# print('索引：',s.index)
# print('数组值：',s.values)
# print('位置下标：',s[2])
# print('标签下标：',s['a'])
# print('位置切片：',s[1:3])
# print('标签切片：',s['a':'c'])
# print('位置列表存取元素：',s[[1,2,3]])#按照标签的位置
# print('标签数组存取元素：',s[['b','c','d']])
# print('字典功能：',list(s.iteritems()))
# s2=pd.Series([20,30,40,50,60],index=['b','c','d','e','f'])
# print(s+s2)#两边index都有则相加即可，若有一方没有，则用NaN填充

# DataFrame对象
#dates = pd.date_range('20130101', periods=6)#periods为个数
# 调用read_csv读取数据，通过index_col指定第0,1列为索引，用parse_dates参数指定进行日期转换列.在指定列时可以使用列的序号或列明。
#df_soil=pd.read_csv('C:\\Users\\Administrator\\Desktop\\PythonLeaning\\Data-Analysis-And-Mining\\Soils-simple.csv',index_col=[0,1],parse_dates=['Date'])
#df_soil=pd.read_csv('Soils-simple.csv',index_col=[0,1],parse_dates=['Date'])
#df_soil.columns.name='Measures'

# #DataFrame对象是一个二维表格，其中每列中的元素必须一致，而不同列则可不同。object可保存任何python对象
# print(df_soil.dtypes)#dtypes属性获得表示各个列类型的Series对象。
# print(df_soil.shape)#行数和列数
# # DataFrame对象用油行索引和列索引，并可通过索引标签对数据进行存取。
# print('columns:',df_soil.columns)# 列索引
# print('index:',df_soil.index)# 行索引
# print('columnsname:',df_soil.columns.name)
# print('indexname:',df_soil.index.name)
# #运算符可通过列索引标签获取指定列，当下标是单个便签时，所得到的是Series对象，当下标是列表时，则是新的DataFrame对象。
# print('pH:',df_soil['pH'])
# print('Dens,Ca:',df_soil[['Dens','Ca']])
# #通过行索引获取指定的行
# print('df.loc:',df_soil.loc['0-10','Top'])
# print('df.loc:',df_soil.loc[['10-30']])
# print(df_soil.values.dtype)
# print('===============将内存中的数据转换成DataFrame对象===================')
# import numpy as np
#
# # 将形状为4,2的二维数组转换成DataFrame对象，通过index和columns参数指定行和列的索引
# df1 = pd.DataFrame(np.random.randint(0, 10, (4, 2)), index=['A', 'B', 'C', 'D'], columns=['a', 'b'])
# # 将字典转换成DataFrame，列索引有字典的键决定，行索引有index参数指定
# df2 = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]}, index=['A', 'B', 'C', 'D'])
# # 将结构数组转行成DataFrame对象，列索引由结构数组的字段名决定，行索引默认从0开始的整数序列。
# df3 = pd.DataFrame(
#     np.array([('item1', 1), ('item2', 2), ('item3', 3), ('item4', 4)], dtype=[('name', '10S'), ('count', 'int')]))
#
# print(df1)
# print(df2)
# print(df3)
#
# dict1 = {'a': [1, 2, 3], 'b': [4, 5, 6]}
# dict2 = {'a': {'A': 1, 'B': 2}, 'b': {'A': 3, 'C': 4}}
#
# # orient参数指定字典键对应的方向，默认columns，表示把字典的键转换成为列索引
# df1 = pd.DataFrame.from_dict(dict1, orient='index')
# df2 = pd.DataFrame.from_dict(dict1, orient='columns')
# df3 = pd.DataFrame.from_dict(dict2, orient='index')
# df4 = pd.DataFrame.from_dict(dict2, orient='columns')
#
# print(df1)
# print(df2)
# print(df3)
# print(df4)
#
# df1 = pd.DataFrame.from_items(dict1.items(), orient='index', columns=['A', 'B', 'C'])
# df2 = pd.DataFrame.from_items(dict1.items(), orient='columns')
#
# print(df1)
# print(df2)
#
# print("======将DataFrame对象转换成其他格式的数据======")
# print(df2.to_dict(orient='records'))#字典列表
# print(df2.to_dict(orient='list'))#列表列表
# print(df2.to_dict(orient='dict'))#嵌套字典
# print(df2.to_records(index=False).dtype)#转换成结构数组,index表示返回是否行索引数据
# print(df2.to_records(index=True).dtype)#转换成结构数组,index表示返回是否行索引数据
#index=df_soil.columns
# print(index)
# print(index.values)
#
# # Index对象可当做一维数组，可通过NumPy数组相同的下标操作获取新的Index对象，但是只可读，不可改。
# print(index[[1,3]])
# print(index>'c')
# print(index[index>'c'])
# print(index[1::2])
# # 获取映射值
# print(index.get_loc('Ca'))#获取位置
# print(index.get_indexer(['Dens','Conduc']))#以数组排序，若不存在 返回-1
# # 可直接调用Index()创建Index对象，再传递给DataFrame的index或columns，又由于Index不可变，则多个数据可共用一个Index对象
# index=pd.Index(['A','B','C'],name='level')
# s1=pd.Series([1,2,3],index=index)
# df1=pd.DataFrame({'a':[1,2,3],'b':[4,5,6]},index=index)
# print(s1.index,df1.index)

# mindex=df_soil.index
# print(mindex[1])
# print(mindex.get_loc(('0-10','Slope')))
# print(mindex.get_indexer([('10-30','Top')]))
# # 内部并不直接保存元组对象，而是使用多个Index对象保存索引中每级的标签
# print(mindex.levels[0])
# print(mindex.levels[1])
#
# #使用多个整数数组保存标签下标
# print(mindex.labels[0])
# print(mindex.labels[1])
# # 获得所有元组列表
#
# level0,level1=mindex.levels
# label0,label1=mindex.labels
#
# print(list(zip(level0[label0],level1[label1])))
#
# # 将一个元组列表传递给Index（）时，将自动创建MultiIndex对象
# print(pd.Index([('A','x'),('A','y'),('B','x'),('B','y')],name=['class1','class2'],tupleize_cols=True))#tupleize_cols参数表示是否为MultiIndex对象
# print(pd.MultiIndex.from_arrays([['A','A','B','B'],['x','y','x','y']],names=['class1','class2']))#多个数组创建MultiIndex对象
# midx=pd.MultiIndex.from_product([['A','B','C'],['X','Y']],names=['class1','class2'])#使用笛卡尔积创建MultiIndex对象
# print(midx)
# print(pd.DataFrame(np.random.randint(0,10,(6,6)),columns=midx,index=midx))
import numpy as np
# np.random.seed(42)
# df=pd.DataFrame(np.random.randint(0,10,(5,3)),index=['r1','r2','r3','r4','r5'],columns=['c1','c2','c3'])
# print(df)
#print(df[['c1']])
#print(df[['c1','c2']])
#print(df[2:4])#表示第2行到第4-1行
#print(df['r2':'r4'])#表示第2行到第4-1行
#print(df[df.c1>4])#c1列大于4的行
#print(df[df>2])#对象中大于2的显示，否则用NaN填充


# print(df.loc['r2'])#r2行
# print(df.loc['r2','c2'])#第r2行的第c2列
# print(df.loc[['r2','r3']])#r2，与r3行
# print(df.loc[['r2','r3'],['c2','c3']])#r2和r3所对应的c2于c3列
# print(df.loc['r2':'r4',['c2','c3']])#r2到r4所对应的c2，c3列
# print(df.loc[df.c1>2,['c1','c2']])#c1列大于2的c1与c2列

# print(df.iloc[1])#r2行
# print(df.iloc[[1,2]])#r2，与r3行
# print(df.iloc[[1,2],[1,2]])#r2和r3所对应的c2于c3列
# print(df.iloc[1:3,[1,2]])#r2到r4所对应的c2，c3列
# print(df.iloc[df.c1.values>2,[0,1]])#c1列大于2的c1与c2列

# print(df.ix[2:4,['c1','c2']])
# print(df.ix['r1':'r3',[0,1]])

# print(df.at['r2','c2'])
# print(df.iat[1,1])
# print(df.get_value('r2','c2'))#比.at速度要快
# print(df.lookup(['r2','r4','r3'],['c1','c2','c1']))#获取指定元素数组

# print(df_soil.loc[np.s_[:'Top'],['pH','Ca']])

# print(df.query('c1>3 and c2<4'))#条件查询 可使用no、and、or等关键字，可使用全局变量
# low=3
# hi=4
# print(df.query('c1>@low and c2<@hi'))


#数据输入输出

# df_list=[]
# for df in pd.read_csv(
#     '201406.csv',
#     encoding='utf-8',#编码
#     chunksize=100,#一次读入的行数
#     usecols=['时间','监测点','AQI','PM2.5','PM10'],#只读入这些列
#     na_values=['-',''],#这些字符串表示缺失数据
#     parse_dates=[0]#第一列为时间列
#     ):
#     df_list.append(df)
#
# print(df_list[0].count())
# print(df_list[0].dtypes)
#
# #HDF5文件
#
# store=pd.HDFStore('a.hdf5',complib='blosc',complevel=9)#complib指定使用blosc压缩数据，通过complevel指定压缩级别
# df1=pd.DataFrame(np.random.rand(100000,4),columns=list('ABCD'))
# df2=pd.DataFrame(np.random.randint(0,1000,(10000,3)),columns=['one','two','three'])
# s1=pd.Series(np.random.rand(1000))
# store['dataframes/df1']=df1
# store['dataframes/df2']=df2
# store['dataframes/s1']=s1
#
# print(store.keys())
# print(df1.equals(store['dataframes/df1']))
#
# root=store.get_node("//")#获取根节点,
# for node in root._f_walknodes():#调用_f_walknodes()遍历其包含的所有节点。
#     print(node)
#
# #往已保存进HDFStore的DataFrame追加数据
# store.append('dataframe/df_dynamic1',df1,append=False)#append为False表示是否覆盖已存在的数据
#
# df3=pd.DataFrame(np.random.rand(100,4),columns=list('ABCD'))
# store.append('dataframe/df_dynamic1',df3)
# print(store['dataframes/df_dynamic1'].shape)
#
# #使用append()创建pytables中支持索引的表格（Table）节点，默认使用DataFrame的index作为索引。通过select可对其进行查询 index表DataFrame的标签数据
# print(store.select('dataframe/df_dynamic1',where='index>97 & index<102'))#表示行在97-102所有的行，该方法减少内存使用量和磁盘读取速度以及数据的访问速度,在添加df3时是添加不是覆盖，所以有两个98,99
#
# #若希望对DataFrame的指定列进行索引，可以在用append创建新的表格时，通过data_columns指定索引列
# store.append('dataframes/df_dynamic1', df1, append=False, data_columns=['A', 'B'])
# print(store.select('dataframes/df_dynamic1' , where='A > 0.99 & B <0.01'))


#读写数据库

# Pickle序列化

df = pd.DataFrame(np.random.randint(0, 10, (4, 2)), index=['A', 'B', 'C', 'D'], columns=['a', 'b'])
df.to_pickle('df.pickle')
df_aqi2=pd.read_pickle('df.pickle')
print(df.equals(df_aqi2))

