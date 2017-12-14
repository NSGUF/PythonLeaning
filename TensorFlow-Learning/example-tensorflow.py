# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:39:04 2017

@author: ZhifengFang
"""
#Tensorflow中的所有计算都会被转化为计算图上的节点，而节点之间的边描述了
#计算之间的依赖关系
import tensorflow as tf #先引用包
#print(tf.__version__)#查看版本
#hello=tf.constant('Hello')#一个计算，计算结果为一个张量，保存在变量hello中
#a=tf.constant([1.0,2.0],name='a')
#b=tf.constant([2.0,3.0],name='b')
#g=tf.Graph()#获取整个图
#print(a.graph is tf.get_default_graph())#a.graph用于查看张量a所属的计算图
#==============================================================================
# with g.device('/gpu:0'):#指定计算运行的设备
#     result=a+b
#==============================================================================
#result=a+b#a和b的类型必须一样，否则会报错，所以建议在创建变量时，在后边添加dtype=tf.float32
#print(result)
#print(result.get_shape())#获取维度
#tf.InteractiveSession()加载它自身作为默认构建的session，tensor.eval()和operation.run()取决于默认的session.
#换句话说：InteractiveSession 输入的代码少，原因就是它允许变量不需要使用session就可以产生结构。
#会话总结
#==============================================================================
# sess=tf.InteractiveSession()#方法一，这是官网上给的方法
# print(result.eval())
# sess.close()
#==============================================================================
#==============================================================================
# sess=tf.Session()#方法二，close方法必须调用，否则造成内存泄漏
# print(sess.run(result))#该方法与下行结果相同
# print(result.eval(session=sess))#在方法一中，则无需指定session
# sess.close()
#==============================================================================
#==============================================================================
# with tf.Session() as sess:#改编方法二、利用上下文管理器解决close的调用，
#     print(sess.run(result))
# sess=tf.Session()#Tensorflow会自动生成一个默认的计算图，若没有特殊指定，则运算自动加入，但tensorflow不会自动生成默认的会话
# with sess.as_default():#指定默认会话
#     print(result.eval())
#==============================================================================

#==============================================================================
# #使用ConfigProto可以对会话进行配置，且对上述两种方法都可
# config=tf.ConfigProto(allow_soft_placement=True,
#                       log_device_placement=True)#第一个参数表示当在某些不能在GPU上运算的情况下，自动调整到CPU上，第二个参数表示日志中将会记录每个节点在哪个设备上以方便调试，默认都为false
# sess1=tf.InteractiveSession(config=config)
# sess2=tf.Session(config=config)
# 
# with sess1.as_default():
#     print(result.eval())
# 
#==============================================================================



#==============================================================================
# g1=tf.Graph()
# with g1.as_default():
#     v=tf.get_variable('v',[1],initializer=tf.zeros_initializer)
# 
# g2=tf.Graph()
# with g2.as_default():
#     v=tf.get_variable('v',[1],initializer=tf.ones_initializer)
#     
# with tf.Session(graph=g1) as sess:
#     tf.global_variables_initializer().run()#初始化变量
#     with tf.variable_scope("",reuse=True):#管理变量 reuse表示共享
#         print(sess.run(tf.get_variable('v')))
# 
# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope('',reuse=True):
#         print(sess.run(tf.get_variable('v')))
#==============================================================================

#==============================================================================
# #三层简单神经网络
# w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))#声明变量的方法，声明了之后类型不可变，shape可变，使用validate_shape=False参数，但是使用罕见
# w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# x=tf.constant([[0.7,0.9]])
# #定义向前传播的神经网络
# a=tf.matmul(x,w1)#矩阵乘法
# y=tf.matmul(a,w2)
# #==============================================================================
# # sess=tf.Session()
# # sess.run(w1.initializer)#创建的变量都要初始化，给变量赋值
# # sess.run(w2.initializer)
# # print(sess.run(y))#只需run最后结果即可
# # sess.close()
# #==============================================================================
# with tf.Session() as sess:
#     #sess.run(w1.initializer)
#     #sess.run(w2.initializer)
#     tf.global_variables_initializer().run()#初始化所有变量，和上面方法结果一样
#     print(sess.run(y))#只需run最后结果即可
#==============================================================================

#==============================================================================
# #利用placeholder增加输出
# w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))#声明变量的方法，stddev：标准差，mean：平均值，seed:随机数种子
# w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# x=tf.placeholder(tf.float32,shape=(3,2),name='input')
# #定义向前传播的神经网络
# a=tf.matmul(x,w1)#矩阵乘法
# y=tf.matmul(a,w2)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(y, feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))#增加多个输出 feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]},需要和placeholder中的shape值对应
# print(tf.all_variables())#获取计算图上所有的变量
# print(tf.trainable_variables())#获取需要优化的参数
# 
# #Tensorflow随机数生成函数
# tf.random_normal()#正态分布  主要参数：平均值，标准差，取值类型
# tf.truncated_normal()#正态分布，若随机值偏离平均值超过两个标准差，则重新随机 主要参数：平均值，标准差，取值类型
# tf.random_uniform()#均匀分布 主要参数：最小最大取值，取值类型
# tf.random_gamma()#Gamma分布 主要参数：形状参数alpha、尺度参数beta、取值类型
# #常数生成函数和NumPy很像
# tf.zeros([2,3],tf.int32)#全0
# tf.ones([2,3],tf.int32)#全1
# tf.fill([2,3],9)#全填满后边的参数9
# tf.constant([1,2,3])#给定值
# #张量的定义：变量的声明函数tf.Variable是一个运算，运算输出结果就是一个张量，所以，变量只是一种特殊的张量
#==============================================================================

from numpy.random import RandomState

batch_size = 8#定义训练数据的大小
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))#定义神经网络的参数
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) 
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值。
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("\n")
    
    # 训练模型。
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 128
        end = (i*batch_size) % 128 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    
    # 输出训练后的参数取值。
    print("\n")
    print("w1:", sess.run(w1))
    print( "w2:", sess.run(w2))










































