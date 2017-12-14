# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:21:39 2017

@author: Fangzhifeng
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集，如果指定位置没有已经下载好的数据，则会自动在网上下载
mnist=input_data.read_data_sets("/MNIST_data/",one_hot=True)
print("训练数据大小：",mnist.train.num_examples)
print("验证数据大小：",mnist.validation.num_examples)
print("测试数据大小：",mnist.test.num_examples)
print("样本数据大小：",mnist.train.images[0])
print("样本数据标签：",mnist.train.labels[0])
print(tf.__version__)
batch_size=100
#从train的集合中选取batch_size个训练数据
xs,ys=mnist.train.next_batch(batch_size)
print(xs.shape,ys.shape)#xs表示100张图片，然后一张图片784个点，ys表示100张图片对应的数

#MNIST数据集相关的常数
INPUT_NODE=784 #输入层节点，784个像素
OUTPUT_NODE=10 #输出层的节点，类别的数目

#配置神经网络的参数
LAYER1_NODE=500 #隐藏层节点数
BATCH_SIZE=100 #训练数据个数
LEARNING_RATE_BASE=0.8 #基础学习率
LEARNING_RATE_DECAY=0.99 #学习率的衰减率
REGULARI2TION_RATE=0.0001 #正则化在损失函数中的系数
TRAINING_STEPS=5000 #训练轮数
MOVING_AVERAGE_DECAY=0.99 #滑动平均衰减率

#定义辅助函数来计算前向传播结果，使用ReLU做为激活函数
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class==None:
        #计算隐藏层的前向传播结果，这里使用了ReLU激活函数
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        #在计算损失函数时会一并计算softmax函数，所以输出层可不用
        return tf.matmul(layer1,weights2)+biases2
    else:
        #先计算变量的滑动平均值，再计算相应的神经网络前向传播结果
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

#定义训练模型过程
def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    
    #生成隐藏层参数
    weights1=tf.Variable(
            tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))#产生正太分布
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))#偏置项
    #生成输出层参数
    weights2=tf.Variable(
            tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))#产生正太分布
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    #计算在当前参数下神经网络前向传播的结果，这里给出的用于计算滑动平均的类为None
    y=inference(x,None,weights1,biases1,weights2,biases2)
    #定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里指定这个变量
    #为不可训练的变量，一般情况下，训练轮数的变量指定为不可训练的参数
    global_step=tf.Variable(0,trainable=False)
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类，
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #在所有代表神经网络参数的变量上使用滑动平均，tf.trainable_variables返回的是图上集合
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    average_y=inference(x,variable_averages,weights1,biases1,weights2,biases2)
    #计算交叉熵及其平均值 tf.argmax(y_,1),得到对应的结果
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    
    regularizer=tf.contrib.layers.l2_regularizer(REGULARI2TION_RATE)
    regularization=regularizer(weights1)+regularizer(weights2)
    loss=cross_entropy_mean+regularization
    learning_rate=tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True
            )
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed={x:mnist.validation.images,
                       y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print(i,validate_acc)
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(TRAINING_STEPS,test_acc)
        
        
def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()













































