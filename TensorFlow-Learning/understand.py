# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:56:49 2017

@author: Administrator
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#参数
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500

BATCH_SIZE=100

MOVING_AVERAGE_DECAY=0.99
# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 5000        
MOVING_AVERAGE_DECAY = 0.99  

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if(avg_class==None):
        layers=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layers,weights2)+biases2
    else:
        layers=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+biases1)
        return tf.matmul(layers,avg_class.average(weights2))+biases2
    
def train(mnist):
    x=tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y-input')
    
    weights1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    weights2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    y=inference(x,None,weights1,biases1,weights2,biases2)
    
                                                                                                                                                                                                                                                                                                                                                                      
    #添加滑动平均、
    global_step=tf.Variable(0,trainable=False)
    
    variable_avgerases=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_avgerases_op=variable_avgerases.apply(tf.trainable_variables())
    
    average_y=inference(x,variable_avgerases,weights1,biases1,weights2,biases2)
    
    #计算交叉熵及平均值
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    
    #计算正则
    regularizer=tf.contrib.layers.l2_regularizer(0.0001)
    regulation=regularizer(weights1)+regularizer(weights2)
    
    loss=cross_entropy_mean+regulation
    
    #指数衰减学习率
    learning_rate=tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    with tf.control_dependencies([train_step,variable_avgerases_op]):
        train_op=tf.no_op(name='train')
    #对比预测和正确结果
    crrocet_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(crrocet_prediction,tf.float32))
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed={x:mnist.validation.images,
                       y_:mnist.validation.labels}
        
        test_feed={x:mnist.test.images,
                   y_:mnist.test.labels}
        for i in range(5000):
            if i%1000==0:
                validate_result=sess.run(accuracy,feed_dict=validate_feed)
                print(i,validate_result)
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_result=sess.run(accuracy,feed_dict=test_feed)
        print(test_result)
            
def main():
    mnist=input_data.read_data_sets('/MNIST_data/',one_hot=True)  
    train(mnist)

if __name__=='__main__':
    main()
    
    
    






































