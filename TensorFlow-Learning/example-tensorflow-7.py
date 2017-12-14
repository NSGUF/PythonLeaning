# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:02:04 2017

@author: Administrator
"""

#==============================================================================
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# 
# #生成整数型的属性
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 
# #生成字符串型的属性
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 
# mnist=input_data.read_data_sets('MNIST_data',dtype=tf.uint8,one_hot=True)
# images=mnist.train.images
# labels=mnist.train.labels
# pixels=images.shape[1]
# num_examples =mnist.train.num_examples
# #输出地址
# filename='output.tfrecords'
# #创建writer写文件
# writer=tf.python_io.TFRecordWriter(filename)
# for index in range(num_examples):
#     image_row=images[index].tostring()
#     example=tf.train.Example(features=tf.train.Features(feature={
#             'pixels':_int64_feature(pixels),
#             'label':_int64_feature(np.argmax(labels[index])),
#             'image_row':_bytes_feature(image_row)
#             }))
#     writer.write(example.SerializeToString())
# writer.close()
#==============================================================================

#==============================================================================
# import tensorflow as tf
# 
# reader = tf.TFRecordReader()
# filename_queue = tf.train.string_input_producer(["output.tfrecords"])
# _,serialized_example = reader.read(filename_queue)
# # 解析读取的样例。
# features = tf.parse_single_example(
#     serialized_example,
#     features={
#         'image_raw':tf.FixedLenFeature([],tf.string),
#         'pixels':tf.FixedLenFeature([],tf.int64),
#         'label':tf.FixedLenFeature([],tf.int64)
#     })
# images = tf.decode_raw(features['image_raw'],tf.uint8)
# labels = tf.cast(features['label'],tf.int32)
# pixels = tf.cast(features['pixels'],tf.int32)
# sess = tf.Session()
# # 启动多线程处理输入数据。
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess,coord=coord)
# for i in range(10):
#     image, label, pixel = sess.run([images, labels, pixels])
# 
#==============================================================================

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#读取图片
image_raw_data = tf.gfile.FastGFile("datasets/cat.jpg",'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
#    print(img_data.eval())
    img_data.set_shape([1797,2673,3])
#    print(img_data.get_shape())
    #打印图片
    plt.imshow(img_data.eval())
    plt.show()
    
    #将数据的类型转化成实数方便下面的样例程序对图像进行处理
#    img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    
#    #将表示一张图片的三维矩阵重新按照jpeg格式编码并存入文件中，打开这张图片
#    encoded_image=tf.image.encode_jpeg(img_data)
#    
#    with tf.gfile.GFile('output','wb') as f:
#        f.write(encoded_image)
    

    #调整图像大小，第一个参数为原始图片，第二个参数为调整后大小，
    #第三个参数表示调整图片大小的算法，有四种算法，分别为0-双线性插值法（Bilinear interpoliation）
    #1-最近邻居法（Nearest neighbor interpolation
    #2-双三次插值法（Bicubic interpolation）
    #3-面积插值法（Area interpolation）
#==============================================================================
#     resized=tf.image.resize_images(img_data,[300,300],method=0)
#     #Tensorflow的函数处理图片后存储的数据时float32格式，需要转换成uint8
#     print(resized.dtype,img_data.get_shape())
#     cat=np.asarray(resized.eval(),dtype='uint8')
#     #img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32) 与上一行对应
#     plt.imshow(cat)
#     plt.show()
#     
#     #裁剪和填充图片
#     #第一个参数是原始图片，后面两个分别是目标图像大小，如果目标图片比原始图片大，则大的
#     #地方用0填充，反之，则截取图片的正中心
#     croped=tf.image.resize_image_with_crop_or_pad(img_data,1000,1000)
#     padded=tf.image.resize_image_with_crop_or_pad(img_data,3000,3000)
#     
#     plt.imshow(croped.eval())
#     plt.show()
#     plt.imshow(padded.eval())
#     plt.show()
#     
#     
#     #通过比例裁剪图片
#     #第一个参数是原图，第二个是比例大小
#     central_cropped=tf.image.central_crop(img_data,0.5)
#     plt.imshow(central_cropped.eval())
#     plt.show()
#     
#     #裁剪或填充指定区域的图片
#     #第二个参数表示距离图片上方的距离，第三个参数表示距离左侧的距离，
#     #最后面两个参数是高和宽
#     #注意：给的原图大小必须大于目标尺寸加上距离的上和左
#     central_cropped=tf.image.crop_to_bounding_box(img_data,0,0,1000,500)
#     plt.imshow(central_cropped.eval())
#     plt.show()
#     
#     #用0去填充图片，第一个参数是原图，第二个是距离上面的距离，第三个是距离左侧的距离，
#     #后面两个是目标图片的高和宽    
#     pad_to_bounding=tf.image.pad_to_bounding_box(img_data,0,300,3000,5000)
#     plt.imshow(pad_to_bounding.eval())
#     plt.show()
#     #图片翻转
#     #将图片上下翻转
#     flipped=tf.image.flip_up_down(img_data)
#     plt.imshow(flipped.eval())
#     plt.show()
#     #左右翻转
#     flipped=tf.image.flip_left_right(img_data)
#     plt.imshow(flipped.eval())
#     plt.show()
#     #对角线翻转
#     flipped=tf.image.transpose_image(img_data)
#     plt.imshow(flipped.eval())
#     plt.show()
#     
#     # 为了让计算机识别不同角度的实体，可以随机翻转。以一定概率上下翻转图片。
#     #flipped = tf.image.random_flip_up_down(img_data)
#     # 以一定概率左右翻转图片。
#     #flipped = tf.image.random_flip_left_right(img_data)
#     
#     #调整图片色彩
#     #改变图片的亮度-1到1
#     adjusted=tf.image.adjust_brightness(img_data,1.5)
#     # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
#     adjusted=tf.image.random_brightness(img_data,max_delta=0.5)
#     
#     #将图片的对比度-5
#     adjusted = tf.image.adjust_contrast(img_data, -5)
#     #将图片的对比度+5
#     adjusted = tf.image.adjust_contrast(img_data, 5)
#     #在[lower, upper]的范围随机调整图的对比度。
#     #adjusted = tf.image.random_contrast(img_data, lower, upper)
#     
#     plt.imshow(adjusted.eval())
#     plt.show()
#     #调整图片的色相
#     adjusted = tf.image.adjust_hue(img_data, 0.1)
#     # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
#     #adjusted = tf.image.random_hue(image, max_delta)
#     
#     # 将图片的饱和度-5。
#     #adjusted = tf.image.adjust_saturation(img_data, -5)
#     # 将图片的饱和度+5。
#     #adjusted = tf.image.adjust_saturation(img_data, 5)
#     # 在[lower, upper]的范围随机调整图的饱和度。
#     #adjusted = tf.image.random_saturation(img_data, lower, upper)
# 
#     # 图像标准化，将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
#     #adjusted = tf.image.per_image_whitening(img_data)
#     
#     plt.imshow(adjusted.eval())
#     plt.show()
#==============================================================================
    
    #添加标注框
#==============================================================================
#     #将图片缩小一些，这样可视化能让标注框更加清楚。
#     img_data=tf.image.resize_images(img_data,[180,267],method=1)
#     #tf.image.draw_bounding_boxes函数要求图像矩阵中的数字为实数，所以需要先将图像
#     #矩阵转换为实数类型，tf.image.draw_bounding_boxes输入的是一个bacth的数据，也就是多张图像组成的四维矩阵，所以需要
#     #将解码后的图像矩阵加一维。
#     batched=tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
#     #给出每一张图片的所有标注框一个标准框有四个数字，分别代表
#     #[y最小，x最小，y最大，x最大]
#     boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
#     #加入标注
#     result=tf.image.draw_bounding_boxes(batched,boxes)
#     
#     plt.imshow(result[0].eval())
#     plt.show()
#==============================================================================
    
    
#==============================================================================
#     boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
#     #可通过提供标注框的方式来告诉随机截取图像的算法有哪些部分的‘有信息量‘
#     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
#         tf.shape(img_data), bounding_boxes=boxes)
# 
#     # 通过标注框可视化随季截取得到的图像。
#     batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
#     image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
# 
#     distorted_image = tf.slice(img_data, begin, size)
#     plt.imshow(distorted_image.eval())
#     plt.show()
#==============================================================================
    
import tensorflow as tf

#==============================================================================
# #操作一个列队
# #创建一个先进先出队列，指定队列中只能保留两个元素，类型为整数
# q = tf.RandomShuffleQueue(capacity=4, min_after_dequeue=2, dtypes="int32")#capacity表总容量，min_after_dequeue表出队后的最小长度，总容量不能小于初始化给的个数，min_after_dequeue不能小于初始化个数减一
# #初始化队列中的元素，使用队列之前必须初始化
# init=q.enqueue_many(([0,10,10,1,2],))
# #将列队中的第一个元素去除并返回其值
# x=q.dequeue()
# y=x+1
# #重新加入列队
# q_inc=q.enqueue([y])
# with tf.Session() as sess:
#     #初始化
#     init.run()
#     for _ in range(5):
#         v,_=sess.run([x,q_inc])
#         print(v)
#==============================================================================
        
#多线程
#==============================================================================
# import threading
# import time
# import numpy as np
# 
# def MyLoop(coord,work_id):
#     while not coord.should_stop():#是否线程退出了
#         if np.random.rand()<0.1:
#             print('停止于',work_id)
#             coord.request_stop()#通知其他线程退出
#         else:
#             print(work_id)
#         time.sleep(1)
#         
# coord=tf.train.Coordinator()
# threads=[threading.Thread(target=MyLoop,args=(coord,i,)) for i in range(5)]
# for thread in threads:
#     thread.start()
#==============================================================================
#==============================================================================
# #声明一个先入先出列队
# queue=tf.FIFOQueue(100,'float')
# #定义一个如对操作
# enqueue_op=queue.enqueue([tf.random_normal([1])])
# #创建多个线程运行队列的如对操作，启动5个线程，每个线程运行enqueue_op操作
# qr=tf.train.QueueRunner(queue,[enqueue_op]*5)
#  
# #将qr加入到默认集合tf.GraphKeys.QUEUE_RUNNERS集合中
# tf.train.add_queue_runner(qr)
# #定义出队操作
# out_tensor=queue.dequeue()
# with tf.Session() as sess:
#     coord=tf.train.Coordinator()
#     #启动tf.train.add_queue_runner中所有的QueueRunner
#     threads=tf.train.start_queue_runners(sess=sess,coord=coord)
#     for _ in range(3): print(out_tensor.eval())
#     
#     #停止所有线程
#     coord.request_stop()
#     coord.join(threads)
# 
#==============================================================================
#==============================================================================
# #生成样例数据
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 
# num_shards=2#总文件个数
# instances_per_shard=2#每个文件中的数据个数
# 
# for i in range(num_shards):
#     filename=('data.tfrecords-%.5d-of-%.5d' % (i,num_shards))
#     writer=tf.python_io.TFRecordWriter(filename)
#     
#     for j in range(instances_per_shard):
#         exam=tf.train.Example(features=tf.train.Features(feature={
#                 'i':_int64_feature(i),
#                 'j':_int64_feature(j)
#                 }))
#         writer.write(exam.SerializeToString())
#     writer.close()
#==============================================================================

#==============================================================================
# files = tf.train.match_filenames_once("data.tfrecords-*")
# filename_queue = tf.train.string_input_producer(files, shuffle=True,num_epochs=111) 
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)
# features = tf.parse_single_example(
#       serialized_example,
#       features={
#           'i': tf.FixedLenFeature([], tf.int64),
#           'j': tf.FixedLenFeature([], tf.int64),
#       })
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
# #    print(sess.run(files))
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     for i in range(2):
#         print(sess.run([features['i'], features['j']]))
#     coord.request_stop()
#     coord.join(threads)
#==============================================================================

files = tf.train.match_filenames_once("data.tfrecords-*")
filename_queue = tf.train.string_input_producer(files, shuffle=True,num_epochs=111)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          'i': tf.FixedLenFeature([], tf.int64),
          'j': tf.FixedLenFeature([], tf.int64),
      })
example, label = features['i'], features['j']
batch_size = 2
capacity = 1000 + 3 * batch_size
capacity = 1000 + 3 * batch_size
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)

    
    































