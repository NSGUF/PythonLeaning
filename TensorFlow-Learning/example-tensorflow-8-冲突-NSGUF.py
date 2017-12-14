# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:16:42 2017

@author: ZhifengFang
"""

import tensorflow as tf

#操作一个列队

#创建一个先进先出队列，指定队列中只能保留两个元素，类型为整数
q = tf.RandomShuffleQueue(capacity=4, min_after_dequeue=2, dtypes="int32")#(10000, 1000, tf.float32, shapes=[32,32], name='experience_replay')
#初始化队列中的元素，使用队列之前必须初始化
init=q.enqueue_many(([0,10,10,11,34],))
#将列队中的第一个元素去除并返回其值
x=q.dequeue()
y=x+1
#重新加入列队
q_inc=q.enqueue([y])
with tf.Session() as sess:
    #初始化
    init.run()
    for _ in range(5):
        v,_=sess.run([x,q_inc])
        print(v)
        
        
        

