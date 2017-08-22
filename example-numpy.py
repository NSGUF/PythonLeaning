# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:31:11 2017

@author: Administrator
"""

#Numpy（Numerical Python）食高性能科学计算和数据分析的基础包
#NumPy的ndarray：多维数组对象，快速灵活的大数据集容器
#所有元素必须是同类型
#官方文档https://docs.scipy.org/doc/numpy/reference/routines.html
import numpy as np
print(np.version.version)#numpy的版本

data1=[1,2,3,4]
arr1=np.array(data1)
print(arr1)
data2=[[1,2,3,4],[5,6,7,8]]
arr2=np.array(data2)
print(arr2)

print("------------属性---------------")
print(arr2.ndim)#数组的维数
print(arr2.shape)#数组的维度
print(arr2.size)#数组的元素总个数
print(arr2.dtype)#数组中元素的类型
print(arr2.itemsize)#数组中每个元素的字节大小
print(arr2.data)#实际数组元素的缓冲区
#NumPy自带了很多数据类型比如int16,int32等，取值时可用np.int16
#NumPy的转换方式为
print(np.int32(12.123))
print("------------创建数组的方法--------------")
print(np.ones((2,3,4),dtype=np.int16))#全为1的多维数组
print(np.empty((2,3)))#为空的多位数组
print(np.random.random((2,3)))#生成shape为2,3的随机数组
print(np.arange(10,30,5))#以10开始差值为5的等差数列，不包含30
print(np.linspace(10,30,4))#第一个数为10，最后一个数为30，取5个等差数 
print(np.arange(4))#相当下面的式子
print(np.arange(0,4,1))
print(np.arange(12).reshape(4,3))#更换成shape为4,3
a=np.arange(24).reshape(2,3,4)
#print(a.resize(24))#resize作用与reshape相同，但是调用resize将会改变自身，reshape不会
b=np.arange(4)
print(a)#shape为2,3,4，ps：总个数必须为shape的乘积，否则报错
print(a[1,...])#等同于a[1]
print(a[1])#第2个整列
print(a[0,1])#取第一整列的第三列
print(a[1,1,1])#第2大列的第2行的第2列，一个元素
print("------------对数组进行操作--------------")
c=a-b#加减乘除都可，将元素分别对应加减乘除，shape必须对应，除的话注意除数为0的情况
print(c)
print(a**2)#会对数组每个元素进行处理，也可进行一下方法遍历
print(a.sum())#总和
print(a.min())#最小的数
print(a.max())#最大的数
print(a.cumsum())#获取每个数的前n的和，比如前1个位0，前2个位0+1，前3个位0+1+2以此类推
d=np.arange(12).reshape(3,4)
print(d.sum(axis=-2))#总和
print(d.min(axis=0))#最小的数
print(d.max(axis=-1))#最大的数
print(d.mean())#算术平均数
print(d.std())#标准差
print(d.var())#方差
print(d.argmin())#最小索引
print(d.argmax())#最大索引
print(d.cumsum(axis=1))#获取每个数的前n的和，比如前1个位0，前2个位0+1，前3个位0+1+2以此类推
print(d.cumprod())#所有元素的累积积
#总结：当没有axis参数的时候，默认为全部元素，当值为0或-2时，表示每个列中的第一个数，当值为1或-1时，表示以列做运算，其他值则报错
s=np.array([4,3,1,45,2,1,23])
s.sort()#排序
print(np.unique(s))#找出唯一值并排序
print("------------布尔值数组方法--------------")
bools=np.array([False,False,True,False])
print(bools.any())#是否存在一个或多个True
print(bools.all())#全是True
print(a.ravel())#将shape值设为(元素总个数)，相当于a.reshape(a.size)
print(a.ravel().shape)
print(np.sin(a))
print(np.floor(a))#向上取整
print(np.vstack((np.array([[1,1],[1,1]]),np.array([[1,0],[0,1]]))))#合并
print(np.hstack((np.array([[1,1],[1,1]]),np.array([[1,0],[0,1]]))))#合并后翻转
c=a.copy()#深拷贝
print(c is a)
print(a)
print("------------对数组进行遍历--------------")
for row in a:#遍历行
    print(row)
for element in a.flat:#遍历每个元素
    print(element)
print("------------数组文件输入输出--------------")
np.save("save_a.npy",a)#将数组以.npy扩展名存入磁盘
np.load("save_a.npy")
np.savez('array_a_b.npz',a=a,b=b)#将a和b一起存储
arch=np.load('array_a_b.npz')
print(arch['b'])#获取b数组
np.savetxt("text_b.txt",b,delimiter=',')#写入txt文件，以，为分隔符
arr=np.loadtxt("text_a.txt",delimiter=',')#读取txt文件，以，为分隔符
print(arr)
print("----------------------线性代数-------------------------")
print(np.dot(a,b))#矩阵相乘
print(a.transpose())#转置
print(np.trace(a))#迹，对角线的和
e=np.array([[1,2,3],[1,2,3],[1,2,3]])
print(np.diag(e))#以一维数组的形式返回方阵的对角线元素
print("----------------------随机数生成-------------------------")
print(np.random.normal(size=(4,4)))#正态分布随机数，生成4*4的数组，和python内置的random相比，np.random快了很多
print(np.random.binomial(10,0.5,100))#n=10,p=0.5的二项分布
print(np.random.shuffle(b))#对序列随机排序
print(np.random.rand(3,2))#均匀分布的样本值
print(np.random.randint(5,size=(4,3)))#第一个属性为上限表示0-5，
print(np.random.randn())#正态分布（平均值为0，标准差为1）的样本值
print(np.random.beta(10,10,size=10))#beta分布































