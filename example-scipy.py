# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:23:05 2017

@author: ZhifengFang
"""

#SciPy函数库在NumPy库的基础上增加了很多数学、科学以及工程计算中常用的库函数
#比如线性代数、常微分方程数值求解、信号处理、图像处理、稀疏矩阵等
#==============================================================================
# from scipy import constants as C
# print("--------------常数和特殊函数------------------")
# print(C.c)#光速
# print(C.h)#普朗克常数
# print(C.mile)#1英里等于多少米
# print(C.inch)#1英寸等于多少米
# print(C.gram)#1克等于多少千克
# print(C.pound)#1磅等于多少kg
# #SciPy中的special模块是一个非常完整的函数库
# import scipy.special as S
# print(S.gamma(4))#计算伽玛函数
# print(S.gammaln(1000))#计算更大范围
# 
# print("--------最小二乘法拟合直线-----------")
# import numpy as np
# from scipy.optimize import leastsq
# 
# x=np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
# y=np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])
# def residuals(p):
#     k,b=p
#     return y-(k*x+b)
# 
# r=leastsq(residuals,[1,0])
# k,b=r[0]
# print("k=%d,b=%d" %(k,b))
# 
#==============================================================================

print("-----------------文件输入/输出：scrip.io---------------------")
from scipy import io as spio
import numpy as np
#==============================================================================
# a=np.ones((3,3))
# spio.savemat('file.mat',{'a':a})#保存数据
# data=spio.loadmat('file.mat',struct_as_record=True)#获取数据
# print(data['a'])
#==============================================================================











