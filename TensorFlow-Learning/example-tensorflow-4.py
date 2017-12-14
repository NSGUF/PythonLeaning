# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:39:14 2017

@author: ZhifengFang
"""
#第四章 深层神经网络

import tensorflow as tf

#==============================================================================
# #常用优化方法
# tf.train.MomentumOptimizer
# tf.train.GradientDescentOptimizer
# tf.train.AdamOptimizer
#==============================================================================
#==============================================================================
# #激活函数：非线性函数，增加偏置项b
# a=tf.nn.relu(tf.matmul(x,w1)+b1) #增加激活函数和偏置项
# y=tf.nn.relu(tf.matmul(a,w2)+b2)
# #感知机：单层神经网络，没有隐藏层,不能解决异或运算，但多层可以
# #损失函数（loss function）:
# #分类问题和回归问题的经典损失函数，其中分类问题常用的是交叉熵，回归问题常用的是均方误差
# #交叉熵刻画了两个概率分布之间的距离，距离越小越优
# #交叉熵的计算
# cross_entropy=-tf.reduce_mean(#y_表示正确结果，y表示预测结果，tf.clip_by_value函数将y中的值小于1e-10的换成1e-10，大于1.0的换成1.0
#         y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))#*不是矩阵乘法，就是对应的位置相乘，tf.reduce_mean结果为整个矩阵的平均
# #交叉熵一般又与softmax一起用，所以合并为：softmax回归之后的交叉熵损失函数
# cross_entropy=tf.nn.softmax_cross_entropy_with_logits(y,y_)
# #对于回归问题，最常用的损失函数是均方误差
# mse=tf.reduce_mean(tf.square(y_-y))
#==============================================================================
#==============================================================================
# with tf.Session()  as sess:
#     print(tf.reduce_sum([1,2,3,4,5]).eval())#和
#     v1=tf.constant([1,2,3,4])
#     v2=tf.constant([4,3,2,1])
#     print(tf.greater(v1,v2).eval())#v1和v2的每个对应元素比较，若v1大于v2，则为True否则为False
#     print(tf.where(tf.greater(v1,v2),v1,v2).eval())#根据第一个参数的T和F来选择v1和v2，若为False则元素选V2中的值，整个函数相当于选取v1和v2中较大的数组成一个举证
# 
#==============================================================================
#==============================================================================
# from numpy.random import RandomState
# 
# batch_size=8
# x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
# y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')
# w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
# y=tf.matmul(x,w1)
# 
# loss_less=1#重要的属性
# loss_more=10
# #均方差计算损失函数
# loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
# train_step=tf.train.AdamOptimizer(0.001).minimize(loss)#定义反向传播算法来优化神经网络参数
# #通过随机数生成一个模拟数据集
# rdm=RandomState(1)
# X=rdm.rand(128,2)
# Y=[[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]
# with tf.Session() as sess:
#     init_op=tf.global_variables_initializer()
#     sess.run(init_op)
#     STEPS=5000
#     for i in range(STEPS):
#         start=(i*batch_size)%128
#         end=(i*batch_size)%128+batch_size
#         sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#         if i%1000==0:
#             print(i,':',sess.run(w1))
#     print(sess.run(w1))
#==============================================================================

#==============================================================================
# #通过反向传播算法和梯度下降算法调整神经网络中参数的取值
# #学习率的设置
# TRAINING_STEPS=100
# global_step=tf.Variable(1)#初始化为0
# #def exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
# #                      staircase=False, name=None):分别为：初始学习率，经过decay_steps轮计算，学习率乘以decay_rate，staircase为T，学习率阶梯状衰减，否则，连续衰减
# LEA_RATE=tf.train.exponential_decay(0.1,global_step,1,0.96,staircase=False)#获得学习率，指数衰减法，开始较大的衰减率，后来指数变小
# x=tf.Variable(tf.constant(5,dtype=tf.float32),name='x')
# y=tf.square(x)
# train_op=tf.train.GradientDescentOptimizer(LEA_RATE).minimize(y,global_step=global_step)#实现梯度下降算法的优化器，global_step会自动加1，可当计数器用
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(TRAINING_STEPS):
#         sess.run(train_op)
# 
#         LEA_RATE_value=sess.run(LEA_RATE)
#         x_value=sess.run(x)
#         print(i+1, i+1, x_value, LEA_RATE_value,sess.run(global_step))
#==============================================================================

#过拟合问题：模型完全记住了所有训练数据的结果从而得到十分小的损失函数，但是对未知数据可能无法做出可靠的判断
#为避免过拟合，可使用正则化方法，在损失函数中加入刻画模型复杂程度的指标。
#常用两种正则化方式：L1、L2，也可将L1和L2结合起来使用
#==============================================================================
# w=tf.constant([[1.0,-2.0],[-3.0,4.0]])
# with tf.Session() as sess:
#     print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(w)))#w中每个元素的绝对值之和乘0.5，所以值为5
#     print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(w)))#w中每个元素的平方和除以2再乘0.5，结果为7.5
#     print(sess.run(tf.contrib.layers.l1_l2_regularizer(0.5)(w)))#w中的每个元素之和乘0.5+w中每个元素的平方和乘（1-0.5）
#==============================================================================

#==============================================================================
# import matplotlib.pyplot as plt
# import numpy as np
# print(tf.__version__)
# data=[]
# label=[]
# np.random.seed(0)#设置种子，使得下面获取的数据每次都相同
# for i in range(150):
#     x1=np.random.uniform(-1,1)#从-1到1取随机数
#     x2=np.random.uniform(0,2)
#     data.append([np.random.normal(x1,0.1),np.random.normal(x2,0.1)])#np.random.normal(x1,0.1)表示到x1-0.1到x1+0.1随机取一个数
#     if x1**2+x2**2<1:
#         label.append(0)
#     else:
#         label.append(1)
# data=np.hstack(data).reshape(-1,2)#np.array(data)#将list转成数组
# label=np.hstack(label).reshape(-1,1)#np.array(label)
# plt.scatter(data[:,0],data[:,1],c=label,vmin=-.2,vmax=2,edgecolor='white')
# plt.show()
# 
# def get_weight(shape,lambdal):#定义一个获取权重，并自动加入正则项到损失的函数
#     var =tf.Variable(tf.random_normal(shape),dtype=tf.float32)
#     tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambdal)(var))
#     return var
# 
# #定义神经网络
# x=tf.placeholder(tf.float32,shape=(None,2))
# y_=tf.placeholder(tf.float32,shape=(None,1))
# sample_size=len(data)
# layer_dimension=[2,10,5,3,1]
# n_layer=len(layer_dimension)
# cur_layer=x
# in_dimension=layer_dimension[0]
# for i in range(1,n_layer):
#     out_dimension=layer_dimension[i]
#     weight=get_weight([in_dimension,out_dimension],0.003)
#     bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
#     cur_layer=tf.nn.elu(tf.matmul(cur_layer,weight)+bias)
#     in_dimension=layer_dimension[i]
# y=cur_layer
# mse_loss=tf.reduce_sum(tf.pow(y_-y,2))/sample_size
# tf.add_to_collection('losses',mse_loss)
# loss=tf.add_n(tf.get_collection('losses'))
# 
# 
# train_op=tf.train.AdamOptimizer(0.001).minimize(mse_loss)
# TS=40000
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for i in range(TS):
#         sess.run(train_op, feed_dict={x: data, y_: label})
#         if i % 2000 == 0:
#             print("After %d steps, mse_loss: %f" % (i,sess.run(mse_loss, feed_dict={x: data, y_: label})))
# 
#     xx,yy=np.mgrid[-1.2:1.2:.01,-0.2:2.2:.01]
#     grid=np.c_[xx.ravel(),yy.ravel()]
#     probs=sess.run(y,feed_dict={x:grid})
#     probs=probs.reshape(xx.shape)
# 
# plt.scatter(data[:,0],data[:,1],c=label,vmin=-.2,vmax=2,edgecolor='white')
# plt.contour(xx,yy,probs,levels=[.5],cmap="Greys",vmin=0,vmax=.1)
# plt.show()
#==============================================================================
print(tf.__version__)

#==============================================================================
# #滑动平均模型
# v1=tf.Variable(0,dtype=tf.float32)
# step=tf.Variable(0,trainable=False)
# #定义一个滑动平均模型的类，初始化衰减率为0.99，控制衰减率的变量step
# ema=tf.train.ExponentialMovingAverage(0.99,step)
# #定义一个更新变量滑动平均的操作，并给定一个列表，每次执行这个操作时，列表更新
# maintain_ave_op=ema.apply([v1])
# 
# with tf.Session() as sess:
#     init_op=tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run([v1,ema.average(v1)]))#上面的V1*衰减率+（1-衰减率）*更新变量v1，其中衰减率=min{初始化衰减率,（1+step）/（10+step）}
#     
#     sess.run(tf.assign(v1,5))
#     sess.run(maintain_ave_op)
#     print(sess.run([v1,ema.average(v1)]))
#     
#     sess.run(tf.assign(step,10000))
#     sess.run(tf.assign(v1,10))
#     sess.run(maintain_ave_op)
#     print(sess.run([v1,ema.average(v1)]))
#     
#     sess.run(maintain_ave_op)
#     print(sess.run([v1,ema.average(v1)]))
#==============================================================================

