# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:51:23 2017

@author: Administrator
"""

import tensorflow as tf

#==============================================================================
# # 创建过滤器的权重变量，前两个维度代表过滤器的尺寸，
# # 第三个维度表示当前层的深度，第四个维度表示过滤器的深度
# filter_weight=tf.get_variable(
#         'weights',
#         [5,5,3,16],
#         initializer=tf.truncated_normal_initializer(stddev=0.1))
# #过滤器的深度也就是神经网络中下一层节点矩阵的深度
# biases=tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.1))
# #tf.nn.conv2d提供了一个非常方便的函数来实现卷积层前向传播的算法。
# #这个函数的第一个输入为当前当前层的节点矩阵。注意这个矩阵是一个四维矩阵，
# #后面三个维度对应一个节点举证，第一个为对应一个输入batch。
# #比如在输入层，input[0,,,]表示第一章图片，input[1,,,]表第二张图片
# #tf.nn.conv2d的第二个参数表示卷积层的权重，第三个参数表示不同维度上的步长，最后一个
# #维度表示参数填充的方法，SAME表示0填充，AVLID表示不添加
# conv=tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding="SAME")
# #给每个节点加上偏置项
# bias=tf.nn.bias_add(conv,biases)
# #去线性化
# actived_conv=tf.nn.relu(bias)
# #步长的第一维和最后一维只能是1，最大池化层的方法,ksize是过滤器的尺寸，strides是步长，
# pool=tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
# #步长的第一维和最后一维只能是1，平均池化层的方法
# pool=tf.nn.avg_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
#==============================================================================


import tensorflow as tf 
import numpy as np
M = np.array([ [[1],[-1],[0]], [[-1],[2],[1]], [[0],[2],[-2]] ])

filter_weight = tf.get_variable('weights', [2, 2, 1, 1], initializer = tf.constant_initializer([
                                                                        [1, -1],
                                                                        [0, 2]]))
biases = tf.get_variable('biases', [1], initializer = tf.constant_initializer(1))


M = np.asarray(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)

x = tf.placeholder('float32', [1, None, None, 1])
conv = tf.nn.conv2d(x, filter_weight, strides = [1, 2, 2, 1], padding = 'SAME')
bias = tf.nn.bias_add(conv, biases)
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias,feed_dict={x:M})
    pooled_M = sess.run(pool,feed_dict={x:M})
    
    print("convoluted_M: \n", convoluted_M)
    print("pooled_M: \n", pooled_M)









