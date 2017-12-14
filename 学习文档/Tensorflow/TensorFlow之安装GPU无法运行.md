&nbsp;&nbsp;&nbsp;&nbsp;在根据教程<http://blog.csdn.net/sb19931201/article/details/53648615>安装好全部的时候，却无情的给我抛了几个错：  
1、AttributeError: module 'tensorflow' has no attribute 'device'  
&nbsp;&nbsp;&nbsp;&nbsp;这貌似是我先pip了tensorflow-gpu的包，再添加cuDnn库，好吧，重新来过。   
2、ImportError: Could not find 'cudart64\_80.dll'. TensorFlow requires that this DLL be installed in a directory that is named in your %PATH% environment variable. Download and install CUDA 8.0 from this URL: https://developer.nvidia.com/cuda-toolkit   
&nbsp;&nbsp;&nbsp;&nbsp;在下载的CUDA的时候，随手下了9.0的，结果只支持8.0.好吧，重新来过。  
3、ImportError: Could not find 'cudnn64\_6.dll'. TensorFlow requires that this DLL be installed in a directory that is named in your %PATH% environment variable. Note that installing cuDNN is a separate step from installing CUDA, and this DLL is often found in a different directory from the CUDA DLLs. You may install the necessary DLL by downloading cuDNN 6 from this URL: https://developer.nvidia.com/cudnn  
&nbsp;&nbsp;&nbsp;&nbsp;明明看教程的时候写的是5.1版本，我也下的5.1啊，这是为什么？原来是因为我的tensorflow-gpu的版本高于1.3,所以用6.0，好吧，又重新来过。  
4、InvalidArgumentError (see above for traceback): Cannot assign a device for operation 'add': Operation was explicitly assigned to /device:GPU:0 but available devices are [ /job:localhost/replica:0/task:0/device:CPU:0 ]. Make sure the device specification refers to a valid device.
         [[Node: add = Add[T=DT_FLOAT, _device="/device:GPU:0"](a, b)]]  
&nbsp;&nbsp;&nbsp;&nbsp;我的天，这什么鬼东西？我在公司都已经做了一遍的，都没事！还给我弹了下面这个框![](https://i.imgur.com/GaYPEoP.png)  
难道我的显卡坏了？看了下，没坏。好吧，根据下面改就行。
桌面上空白地方右键，进入NVIDIA面板，然后下图
![](https://i.imgur.com/sR7xs9D.png)  
选择第二个，点击应用，再重启电脑即可，记得重启电脑。  
我用的以下代码测试：  

	import tensorflow as tf

	# # 通过tf.device将运算指定到特定的设备上。
	with tf.device('/cpu:0'):
	    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
	    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
	with tf.device('/gpu:0'):
	     c=a+b
	# 通过log_device_placement参数来记录运行每一个运算的设备。
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	print(sess.run(c))
