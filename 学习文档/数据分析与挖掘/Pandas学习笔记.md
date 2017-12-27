# Pandas基础篇
>Pandas基于Numpy开发，提供了很多高级的数据处理功能。
## 1、Pandas中的数据对象
>Series和DataFrame是Pandas中最常用的两个对象。  
### 1.1 Series对象
>是Pandas中最基本的对象，可用Numpy的数组处理函数直接对Series对象进行处理。支持使用位置、索引标签作为下标存取元素，与字典类似；每一个Series对象实际上由两个数组组成。
    
    print(pd.__version__)
	s=pd.Series([1,3,2,4,5],index=['b','a','c','d','e'])#若index不指定，将自动创建位置下标的索引
	print('索引：',s.index)
	print('数组值：',s.values)
	print('位置下标：',s[2])
	print('标签下标：',s['a'])
	print('位置切片：',s[1:3])
	print('标签切片：',s['a':'c'])
	print('位置列表存取元素：',s[[1,2,3]])#按照标签的位置
	print('标签数组存取元素：',s[['b','c','d']])
	print('字典功能：',list(s.iteritems()))
	s2=pd.Series([20,30,40,50,60],index=['b','c','d','e','f'])
	print(s+s2)#两边index都有则相加即可，若有一方没有，则用NaN填充
	print(s.value_counts())#元素中每个值得个数
	str=pd.Series(['A','b','C','是','Baca',np.nan,'VASD','dafd'])
	print(str.str.lower())# 对英文字母小写
    
### 1.2 DataFrame 对象
>DataFrame对象（数据表）是Pandas中最常用的数据对象。Pandas提供了很多数据结构转换成DataFrame对象的方法，还提供很多输入输出函数来将各种文件格式转换成DataFrame对象。

    dates = pd.date_range('20130101', periods=6)#periods为个数
	调用read_csv读取数据，通过index_col指定第0,1列为索引，用parse_dates参数指定进行日期转换列.在指定列时可以使用列的序号或列明。
	df_soil=pd.read_csv('Soils-simple.csv',index_col=[0,1],parse_dates=['Date'])
	df_soil=pd.read_csv('Soils-simple.csv',index_col=[0,1],parse_dates=['Date'])
	df_soil.columns.name='Measures'
	
	DataFrame对象是一个二维表格，其中每列中的元素必须一致，而不同列则可不同。object可保存任何python对象
	print(df_soil.dtypes)#dtypes属性获得表示各个列类型的Series对象。
	print(df_soil.shape)#行数和列数
	print(df_soil)
	
	print(df_soil.head(1))#查看顶部几条数据
	print(df_soil.tail(1))#查看底部几条数据
	print(df_soil.index)#查看行索引
	print(df_soil.columns)#查看列索引
	print(df_soil.values)#查看底层numpy的数据
	print(df_soil.describe())#快速统计数据
	print(df_soil.T)#行列互换
	print(df_soil.sort_index(axis=0,ascending=False))#根据轴排序,axis为1表横轴，反之纵轴，ascending为True表正序，反之 反序
	print(df_soil.sort_values(by='pH'))#按照值排序，指定某纵轴对应的值，列排序
	
	
	#DataFrame对象用油行索引和列索引，并可通过索引标签对数据进行存取。
	print('columns:',df_soil.columns)# 列索引
	print('index:',df_soil.index)# 行索引
	print('columnsname:',df_soil.columns.name)
	print('indexname:',df_soil.index.name)
	#运算符可通过列索引标签获取指定列，当下标是单个便签时，所得到的是Series对象，当下标是列表时，则是新的DataFrame对象。
	print('pH:',df_soil['pH'])
	print('Dens,Ca:',df_soil[['Dens','Ca']])
	#通过行索引获取指定的行
	print('df.loc:',df_soil.loc['0-10','Top'])
	print('df.loc:',df_soil.loc[['10-30']])
	print(df_soil.values.dtype)
	print('===============将内存中的数据转换成DataFrame对象===================')
	import numpy as np
	
	# 将形状为4,2的二维数组转换成DataFrame对象，通过index和columns参数指定行和列的索引
	df1 = pd.DataFrame(np.random.randint(0, 10, (4, 2)), index=['A', 'B', 'C', 'D'], columns=['a', 'b'])
	# 将字典转换成DataFrame，列索引有字典的键决定，行索引有index参数指定
	df2 = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]}, index=['A', 'B', 'C', 'D'])
	# 将结构数组转行成DataFrame对象，列索引由结构数组的字段名决定，行索引默认从0开始的整数序列。
	df3 = pd.DataFrame(
	    np.array([('item1', 1), ('item2', 2), ('item3', 3), ('item4', 4)], dtype=[('name', '10S'), ('count', 'int')]))
	
	print(df1)
	print(df2)
	print(df3)
	
	dict1 = {'a': [1, 2, 3], 'b': [4, 5, 6]}
	dict2 = {'a': {'A': 1, 'B': 2}, 'b': {'A': 3, 'C': 4}}
	
	# orient参数指定字典键对应的方向，默认columns，表示把字典的键转换成为列索引
	df1 = pd.DataFrame.from_dict(dict1, orient='index')
	df2 = pd.DataFrame.from_dict(dict1, orient='columns')
	df3 = pd.DataFrame.from_dict(dict2, orient='index')
	df4 = pd.DataFrame.from_dict(dict2, orient='columns')
	
	print(df1)
	print(df2)
	print(df3)
	print(df4)
	
	df1 = pd.DataFrame.from_items(dict1.items(), orient='index', columns=['A', 'B', 'C'])
	df2 = pd.DataFrame.from_items(dict1.items(), orient='columns')
	
	print(df1)
	print(df2)
	
	print("======将DataFrame对象转换成其他格式的数据======")
	print(df2.to_dict(orient='records'))#字典列表
	print(df2.to_dict(orient='list'))#列表列表
	print(df2.to_dict(orient='dict'))#嵌套字典
	print(df2.to_records(index=False).dtype)#转换成结构数组,index表示返回是否行索引数据
	print(df2.to_records(index=True).dtype)#转换成结构数组,index表示返回是否行索引数据
    
	np.random.seed(42)
	dates =pd.Index(['r1','r2','r3','r4','r5'],name='level')
	df=pd.DataFrame(np.random.randint(0,10,(5,3)),index=dates,columns=['c1','c2','c3'])
	print(df)
	pieces=[df[:2],df[2:4],df[4:]]
	print(pd.concat(pieces))#连接
	
	left=pd.DataFrame({'key':['foo','foo'],'lval':[1,2]})
	right=pd.DataFrame({'key':['foo','foo'],'rval':[4,5]})
	print(pd.merge(left,right,on='key'))# 将一样key的值对应链接
	left=pd.DataFrame({'key':['foo','bar'],'lval':[1,2]})
	right=pd.DataFrame({'key':['foo','bar1'],'rval':[4,5]})
	print(pd.merge(left,right,on='key'))# 将一样key的值对应链接
	
	print(df.append(df.iloc[3],ignore_index=False))# 添加第4行，ignore_index表示忽略给的行号 自动标从0开始。
	
	print(df[['c1']])
	print(df[['c1','c2']])
	print(df[2:4])#表示第2行到第4-1行
	print(df['r2':'r4'])#表示第2行到第4-1行
	print(df[df.c1>4])#c1列大于4的行
	print(df[df>2])#对象中大于2的显示，否则用NaN填充
	
	df2=df.copy()
	df2['c4']=[1,2,3,4,5]
	print(df2)
	print(df2['c4'].isin([1,2,3]))#筛选c4列中值为1,2,3的为True
	
	df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
	df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])#放大或缩小原df
	# df1.loc[dates[0]:dates[1],'E'] = 1
	print(df1)
	
	print(df1.dropna(how='any'))#删除所有有NaN的行
	print(df1.fillna(value=5))#将所有NaN的值转换成5
	
	print(df1.isnull())#将每个位置的值判断是否nan
	print(df)
	print(df.mean())#每列的平均值
	print(df.mean(1))#每行的平均值
	s=pd.Series([1,3,5,np.nan,6],index=dates)
	print(s)
	print(s.shift(1))#在最前面放1个nan，然后再加入s，长度为原来的长度
	print(df.sub(s,axis='index'))#对应s的行减去s的位置的值。减nan，则成nan
	print(df.apply(np.cumsum))#累积求和，从列的上到下
	print(df.apply(lambda x:x.max()-x.min()))#一列中的最大减去最小
	dfgroup = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
	                   'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
	                   'C' : np.random.randn(8),'D' : np.random.randn(8)})
	print(dfgroup.groupby('A').sum())# 通过A列中的值，求出同样值的和
	print(dfgroup.groupby('A').size())# 通过A列中的值，求出同样值的个数
	
	print(dfgroup.groupby(['A','B']).sum())# 通过A、B列中的值，求出同样值的和
	
	tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
	                    'foo', 'foo', 'qux', 'qux'],
	                    ['one', 'two', 'one', 'two',
	                    'one', 'two', 'one', 'two']]))
	index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
	df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
	print(df[:4])
	df2 = df[:4]
	print(df2.stack())#将列对应加入行，变成原行*列的行数
	print(df2.stack().unstack())#变回来
	print(df2.stack().unstack(1))# 将多级索引中第2个变为列
	print(df2.stack().unstack(0))# 将多级索引中第1个变为列
	
	df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
	                   'B' : ['A', 'B', 'C'] * 4,
	                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
	                   'D' : np.random.randn(12),
	                   'E' : np.random.randn(12)})
	print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))# 将D列数据拿出来，行为A、B排列组合，列为C中的值
	rng = pd.date_range('1/1/2012', periods=100, freq='S')# 时间排100s freq表示单位
	# print(rng)
	
	ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
	print(ts.resample('5Min').sum())#打印5分钟内的总数

### 1.3 Index对象
>Index对象保存索引标签数据，它可快速找到标签对应的整数下标

    index=df_soil.columns
	print(index)
	print(index.values)
	
	# Index对象可当做一维数组，可通过NumPy数组相同的下标操作获取新的Index对象，但是只可读，不可改。
	print(index[[1,3]])
	print(index>'c')
	print(index[index>'c'])
	print(index[1::2])
	# 获取映射值
	print(index.get_loc('Ca'))#获取位置
	print(index.get_indexer(['Dens','Conduc']))#以数组排序，若不存在 返回-1
	# 可直接调用Index()创建Index对象，再传递给DataFrame的index或columns，又由于Index不可变，则多个数据可共用一个Index对象
	index=pd.Index(['A','B','C'],name='level')
	s1=pd.Series([1,2,3],index=index)
	df1=pd.DataFrame({'a':[1,2,3],'b':[4,5,6]},index=index)
	print(s1.index,df1.index)
    
### 1.4 MultiIndex对象    
>MultiIndex对象表示多级索引，继承Index，其中的多级标签采用元组对象表示

    mindex=df_soil.index
    print(mindex[1])
    print(mindex.get_loc(('0-10','Slope')))
    print(mindex.get_indexer([('10-30','Top')]))
    # 内部并不直接保存元组对象，而是使用多个Index对象保存索引中每级的标签
    print(mindex.levels[0])
    print(mindex.levels[1])
    
    #使用多个整数数组保存标签下标
    print(mindex.labels[0])
    print(mindex.labels[1])
    # 获得所有元组列表
    
    level0,level1=mindex.levels
    label0,label1=mindex.labels
    
    print(list(zip(level0[label0],level1[label1])))
    
    # 将一个元组列表传递给Index（）时，将自动创建MultiIndex对象
    print(pd.Index([('A','x'),('A','y'),('B','x'),('B','y')],name=['class1','class2'],tupleize_cols=True))#tupleize_cols参数表示是否为MultiIndex对象
    print(pd.MultiIndex.from_arrays([['A','A','B','B'],['x','y','x','y']],names=['class1','class2']))#多个数组创建MultiIndex对象
    midx=pd.MultiIndex.from_product([['A','B','C'],['X','Y']],names=['class1','class2'])#使用笛卡尔积创建MultiIndex对象
    print(midx)
    print(pd.DataFrame(np.random.randint(0,10,(6,6)),columns=midx,index=midx))
    
    
### 1.5 常用的函数参数    
    
    
<table>
    <tr>
        <td>参数</td>
        <td>常用值</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>axis</td>
        <td>0、1</td>
        <td>运算对应的轴</td>
    </tr>
    <tr>
        <td>level</td>
        <td>整数或索引的级别名</td>
        <td>指定运算对应的级别</td>
    </tr>
    <tr>
        <td>fill_value</td>
        <td>数值</td>
        <td>指定运算中出现的NaN的代替填充值</td>
    </tr>
    <tr>
        <td>skipna</td>
        <td>布尔值</td>
        <td>运算是否跳过NaN</td>
    </tr>
    <tr>
        <td>index</td>
        <td>序列</td>
        <td>指定行索引</td>
    </tr>
    <tr>
        <td>columns</td>
        <td>序列</td>
        <td>指定列索引</td>
    </tr>
    <tr>
        <td>numeric_only</td>
        <td>布尔值</td>
        <td>是否只针对数值进行运算</td>
    </tr>
    <tr>
        <td>func</td>
        <td>可调用函数</td>
        <td>指定回调函数</td>
    </tr>
    <tr>
        <td>inplace</td>
        <td>布尔值</td>
        <td>是否原地更新，若为否，则返回新对象</td>
    </tr>
    <tr>
        <td>encoding</td>
        <td>utf8</td>
        <td>指定文本编码</td>
    </tr>
    <tr>
        <td>dropna</td>
        <td>布尔值</td>
        <td>是否删除包好NaN的行</td>
    </tr>
</table>  

## 2、下标存取
### 2.1 []操作符
> 通过[]操作符对DataFrame对象进行存取时，支持以下5中下标对象：  

	import numpy as np
	np.random.seed(42)
	df=pd.DataFrame(np.random.randint(0,10,(5,3)),index=['r1','r2','r3','r4','r5'],columns=['c1','c2','c3'])
	print(df)

* 单个索引标签：获取标签对应的列，返回一个Series对象。

		print(df[['c1']])

* 多个索引标签：获取以列表、数组（注意不能是元组）表示多个标签对应的列，返回一个DataFrame对象

		print(df[['c1','c2']])
* 整数切片：以整数下标获取切片对应的行。

		print(df[2:4])#表示第2行到第4-1行
* 标签切片：当使用标签作为切片时包含终值。

		print(df['r2':'r4'])#表示第2行到第4-1行
* 布尔数组：获取数组中True对应的行。

		print(df[df.c1>4])#c1列大于4的行
* 布尔DataFrame：将DataFrame对象中False对应的元素设置为NaN。

		print(df[df>2])#对象中大于2的显示，否则用NaN填充

### 2.2 .loc[]和.iloc[]存取器

    print(df.loc['r2'])#r2行
    print(df.loc['r2','c2'])#第r2行的第c2列
    print(df.loc[['r2','r3']])#r2，与r3行
    print(df.loc[['r2','r3'],['c2','c3']])#r2和r3所对应的c2于c3列
    print(df.loc['r2':'r4',['c2','c3']])#r2到r4所对应的c2，c3列
    print(df.loc[df.c1>2,['c1','c2']])#c1列大于2的c1与c2列

>.iloc与.ioc类似，不过它是使用整数下标。

    print(df.iloc[1])#r2行
    print(df.iloc[[1,2]])#r2，与r3行
    print(df.iloc[[1,2],[1,2]])#r2和r3所对应的c2于c3列
    print(df.iloc[1:3,[1,2]])#r2到r4所对应的c2，c3列
    print(df.iloc[df.c1.values>2,[0,1]])#c1列大于2的c1与c2列

>.ix[]存取器，混用标签和位置下标。

    print(df.ix[2:4,['c1','c2']])
    print(df.ix['r1':'r3',[0,1]])
>获取单个值

    print(df.at['r2','c2'])
    print(df.iat[1,1])
    print(df.get_value('r2','c2'))#比.at速度要快
    print(df.lookup(['r2','r4','r3'],['c1','c2','c1']))#获取指定元素数组
>多级标签的存取

    print(df_soil.loc[np.s_[:'Top'],['pH','Ca']])#np.s_对象创建第0轴对应的下标。

>query()方法

    print(df.query('c1>3 and c2<4'))#条件查询 可使用no、and、or等关键字，可使用全局变量
    low=3
    hi=4
    print(df.query('c1>@low and c2<@hi'))

## 3、文件的输入输出

输入输出函数  

<table>
    <tr>
        <td>函数名</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>read_csv()</td>
        <td>从CSV格式的文本文件读取数据</td>
    </tr>
    <tr>
        <td>read_excel()</td>
        <td>从Excel文件读入数据</td>
    </tr>
    <tr>
        <td>HDFStore()</td>
        <td>使用HDF5文件读写数据</td>
    </tr>
    <tr>
        <td>read_sql()</td>
        <td>从SQL数据库的查询结果载入数据</td>
    </tr>
    <tr>
        <td>read_pickle()</td>
        <td>读入Pickle序列化之后的数据</td>
    </tr>
</table>

### 3.1 CSV格式

    df_list=[]
    for df in pd.read_csv(
        '201406.csv',
        encoding='utf-8',#编码
        chunksize=100,#一次读入的行数
        usecols=['时间','监测点','AQI','PM2.5','PM10'],#只读入这些列
        na_values=['-',''],#这些字符串表示缺失数据
        parse_dates=[0]#第一列为时间列
        ):
        df_list.append(df)

    print(df_list[0].count())
    print(df_list[0].dtypes)

	df.to_csv('foo.csv')#写入数据
	df.to_excel('foo.xlsx', sheet_name='Sheet1')#写入数据
	pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])#读取数据

### 3.2 HDF5文件
>HDF5用于存储科学计算数据，支持大于2G的文件。采用二进制格式保存数据，可以对数据进行压缩存储，比文本文件更节省空间，存取也更迅速。

	df.to_hdf('foo.h5','df')#写入数据
	pd.read_hdf('foo.h5','df')#读取数据

    store=pd.HDFStore('a.hdf5',complib='blosc',complevel=9)#complib指定使用blosc压缩数据，通过complevel指定压缩级别
    df1=pd.DataFrame(np.random.rand(100000,4),columns=list('ABCD'))
    df2=pd.DataFrame(np.random.randint(0,1000,(10000,3)),columns=['one','two','three'])
    s1=pd.Series(np.random.rand(1000))
    store['dataframes/df1']=df1
    store['dataframes/df2']=df2
    store['dataframes/s1']=s1

    print(store.keys())
    print(df1.equals(store['dataframes/df1']))

    root=store.get_node("//")#获取根节点,
    for node in root._f_walknodes():#调用_f_walknodes()遍历其包含的所有节点。
        print(node)

    #往已保存进HDFStore的DataFrame追加数据
    store.append('dataframe/df_dynamic1',df1,append=False)#append为False表示是否覆盖已存在的数据

    df3=pd.DataFrame(np.random.rand(100,4),columns=list('ABCD'))
    store.append('dataframe/df_dynamic1',df3)
    print(store['dataframes/df_dynamic1'].shape)

    #使用append()创建pytables中支持索引的表格（Table）节点，默认使用DataFrame的index作为索引。通过select可对其进行查询 index表DataFrame的标签数据
    print(store.select('dataframe/df_dynamic1',where='index>97 & index<102'))#表示行在97-102所有的行，该方法减少内存使用量和磁盘读取速度以及数据的访问速度,在添加df3时是添加不是覆盖，所以有两个98,99

    #若希望对DataFrame的指定列进行索引，可以在用append创建新的表格时，通过data_columns指定索引列
    store.append('dataframes/df_dynamic1', df1, append=False, data_columns=['A', 'B'])
    print(store.select('dataframes/df_dynamic1' , where='A > 0.99 & B <0.01'))
>在将对个文本文件的数据逐次写入HDF5文件时，需要注意以下几点：   
1.HDF5文件不支持Unicode字符串，因此需要对Unicode字符串进行编码，转换为字节字符串。由于文件可能包含UTF-8的BOM，因此在读取文件的头三个字节并与BOM比较。  
2.需要使用dtype参数指定这些数值列的类型为浮点数。  
3.需要为HDF5文件中的字符串列指定最大长度，否则该最大长度将由第一个被添加进HDF5文件的数据对象决定。  
 
**注意：**由于所有从CSV文件读入DataFrame对象的行索引都为缺省值，因此HDF5文件中的数据的行索引并不是唯一的。


### 3.3 Pickle序列化

	df = pd.DataFrame(np.random.randint(0, 10, (4, 2)), index=['A', 'B', 'C', 'D'], columns=['a', 'b'])
	df.to_pickle('df.pickle')
	df_aqi2=pd.read_pickle('df.pickle')
	print(df.equals(df_aqi2))
















