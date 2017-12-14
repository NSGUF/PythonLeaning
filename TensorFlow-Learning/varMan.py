# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:05:24 2017

@author: ZhifengFang
"""

#变量管理器，
import tensorflow as tf
#==============================================================================
# #以下为两种创建变量的方法
# v=tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))#变量名必填
# v=tf.Variable(tf.constant(1.0,shape=[1]),name="v")#变量名可选
# #7种不同的初始化函数
# tf.constant_initializer#将变量初始化给定常亮  参数：常量的取值
# tf.random_normal_initializer#将变量初始化给满足正态分布的随机值   参数：正太分布的均值和标准差
# tf.truncated_normal_initializer#将变量初始化为满足正太分布的随机值，但若随机出来的值偏离均值超过2个标准差，那将重新随机   参数：正太分布的均值和标准差
# tf.random_uniform_initializer#将变量初始化为满足平均分布的随机值  参数：最大值最小值
# tf.uniform_unit_scaling_initializer#将变量初始化为满足平均分布的随机值，但不影响输出数量级的随机值  参数：factor产生随机值时乘以的系数
# tf.zeros_initializer#全为0   参数：变量维度
# tf.ones_initializer#全为1  参数：变量维度
# 
# #在名字为foo的命名空间内创建名字为v的变量
# with tf.variable_scope("foo"):
#     v=tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))
#==============================================================================

#==============================================================================
# #reuse设为True后，只能获取已经创建的变量
# with tf.variable_scope("foo",reuse=True):
#     v=tf.get_variable("v",[1])
#     print(v)
#==============================================================================

#==============================================================================
# #总结：当参数reuse为True时，上下文管理器中的get_variable函数只能获取已经创建过的变量，反之，只能创建新变量若同名，则报错
# #从下面例子可以看出，如果reuse为True的上下文管理器中的其他管理器的reuse一概为True，反之，其他管理器为True，则为Ture，为False，则为False，以此类推
# with tf.variable_scope("foo"):
#     print(tf.get_variable_scope().reuse)
#     with tf.variable_scope('root',reuse=True):
#         print(tf.get_variable_scope().reuse)
#         with tf.variable_scope('bar'):
#             print(tf.get_variable_scope().reuse)
#             with tf.variable_scope('bar1'):
#                 print(tf.get_variable_scope().reuse)
#         print(tf.get_variable_scope().reuse)
#     print(tf.get_variable_scope().reuse)
#==============================================================================
    
#==============================================================================
# #在命名空间内创建的变量的名称都会带上这个命名空间名做前缀
# v1 = tf.get_variable("v", [1])
# print(v1.name)
# 
# with tf.variable_scope("foo",reuse=True):
#     v2 = tf.get_variable("v", [1])
# print(v2.name)
# 
# with tf.variable_scope("foo"):
#     with tf.variable_scope("bar"):
#         v3 = tf.get_variable("v", [1])
#         print(v3.name)
#         
# v4 = tf.get_variable("v1", [1])
# print(v4.name)
#==============================================================================

#==============================================================================
# #通过变量的名称来获取变量
# with tf.variable_scope("",reuse=True):                                                                                                      
#     v5 = tf.get_variable("foo/bar/v", [1])
#     print(v5 == v3)
#     v6 = tf.get_variable("v1", [1])     
#     print(v6 == v4)
#==============================================================================

#持久化代码 

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2
#init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
saver.export_meta_graph("Saved_model/model.ckpt.meta.json",as_text=True)
#==============================================================================
# 
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess, "Saved_model/model.ckpt")
# 
# 
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, "Saved_model/model.ckpt")
#     print(sess.run(result))
# 
# 
# #直接加载持久化的图
# saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
# with tf.Session() as sess:
#     saver.restore(sess, "Saved_model/model.ckpt")
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
# 
# #变量重命名
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")
# saver = tf.train.Saver({"v1": v1, "v2": v2})
#==============================================================================

#==============================================================================
# v = tf.Variable(0, dtype=tf.float32, name="v")
# for variables in tf.global_variables(): print(variables.name)
#     
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_averages_op = ema.apply(tf.global_variables())
# for variables in tf.global_variables(): print(variables.name)
#==============================================================================
#==============================================================================
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     
#     sess.run(tf.assign(v, 10))
#     sess.run(maintain_averages_op)
#     # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
#     saver.save(sess, "Saved_model/model2.ckpt")
#     print(sess.run([v, ema.average(v)]))
#==============================================================================



#==============================================================================
# import tensorflow as tf
# from tensorflow.python.framework import graph_util
# 
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
# result = v1 + v2
# 
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     graph_def = tf.get_default_graph().as_graph_def()
#     output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
#     with tf.gfile.GFile("Saved_model/combined_model.pb", "wb") as f:
#            f.write(output_graph_def.SerializeToString())
# 
# from tensorflow.python.platform import gfile
# with tf.Session() as sess:
#     model_filename = "Saved_model/combined_model.pb"
#    
#     with gfile.FastGFile(model_filename, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
# 
#     result = tf.import_graph_def(graph_def, return_elements=["add:0"])
#     print(sess.run(result))
# 
#==============================================================================

reader=tf.train.NewCheckpointReader('Saved_model/model.ckpt')
all_variable=reader.get_variable_to_shape_map()
for variable_name in all_variable:
    print(variable_name,all_variable[variable_name])
print(1,reader.get_tensor('v1'))










