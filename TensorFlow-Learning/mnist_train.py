# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:16:43 2017

@author: Administrator
"""

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100 
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"


def train(mnist):
    x=tf.placeholder(tf.float32,shape=[None,mnist_inference.INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,shape=[None,mnist_inference.OUTPUT_NODE],name='y-input')
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    variable_ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_ema_op=variable_ema.apply(tf.trainable_variables())
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learing_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,MOVING_AVERAGE_DECAY,staircase=True)
    train_step=tf.train.GradientDescentOptimizer(learing_rate).minimize(loss,global_step)
    
    with tf.control_dependencies([train_step,variable_ema_op]):
        train_op=tf.no_op(name='train')
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(30000):
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            
            if i%1000==0:
                print(step,loss_value)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                
                
def main(argvs=None):
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
    train(mnist)
    
if __name__=='__main__':
    tf.app.run();
    
    
    
    
    
    
    
    
    
    
    
    
    
    




























