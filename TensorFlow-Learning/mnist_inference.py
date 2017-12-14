# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:56:09 2017

@author: fangzhifeng
"""

import tensorflow as tf

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

#将训练和测试分成两个独立的程序，这将使得每个组件更加灵活，
#将前向传播的过程抽象成一个单独的库函数 
def get_weight_variable(shape,regularizer): 
    print(shape)
    weights=tf.get_variable('weights',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights
        
def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        print(weights)
        biases=tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        
    with tf.variable_scope('layer2'):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable('biases',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer,weights)+biases
    return layer2









































