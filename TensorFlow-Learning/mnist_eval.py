# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:04:15 2017

@author: Administrator
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

def evaluate(mnist):
        x=tf.placeholder(tf.float32,shape=[None,mnist_inference.INPUT_NODE],name='x-input')
        y_=tf.placeholder(tf.float32,shape=[None,mnist_inference.OUTPUT_NODE],name='u-input')
        
        y=mnist_inference.inference(x,None)
        
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        emp=tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore=emp.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        
        vali={x:mnist.validation.images,y_:mnist.validation.labels}
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    acc_score=sess.run(acc,feed_dict=vali)
                    print(global_step,acc_score)
                    
            time.sleep(10)
                
                                                                                                                                                             
def main(argv=None):
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
    evaluate(mnist)
    
if __name__=='__main__':
    tf.app.run()