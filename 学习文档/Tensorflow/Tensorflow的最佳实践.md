# Tensorflow的最佳实践
&nbsp;&nbsp;
## 1、变量管理
&nbsp;&nbsp;Tensorflow提供了变量管理机制，可直接通过变量的名字获取变量，无需通过传参数传递数据。方式如下：

	#以下为两种创建变量的方法
	v=tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))#变量名必填
	v=tf.Variable(tf.constant(1.0,shape=[1]),name="v")#变量名可选
	#7种不同的初始化函数
	tf.constant_initializer#将变量初始化给定常亮  参数：常量的取值
	tf.random_normal_initializer#将变量初始化给满足正态分布的随机值   参数：正太分布的均值和标准差
	tf.truncated_normal_initializer#将变量初始化为满足正太分布的随机值，但若随机出来的值偏离均值超过2个标准差，那将重新随机   参数：正太分布的均值和标准差
	tf.random_uniform_initializer#将变量初始化为满足平均分布的随机值  参数：最大值最小值
	tf.uniform_unit_scaling_initializer#将变量初始化为满足平均分布的随机值，但不影响输出数量级的随机值  参数：factor产生随机值时乘以的系数
	tf.zeros_initializer#全为0   参数：变量维度
	tf.ones_initializer#全为1  参数：变量维度
	#在名字为foo的命名空间内创建名字为v的变量
	with tf.variable_scope("foo"):
	    v=tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))
	
	#reuse设为True后，只能获取已经创建的变量
	with tf.variable_scope("foo",reuse=True):
	    v=tf.get_variable("v",[1])
	    print(v)
	
	#总结：当参数reuse为True时，上下文管理器中的get_variable函数只能获取已经创建过的变量，反之，只能创建新变量若同名，则报错
	#从下面例子可以看出，如果reuse为True的上下文管理器中的其他管理器的reuse一概为True，反之，其他管理器为True，则为Ture，为False，则为False，以此类推
	with tf.variable_scope("foo"):
	    print(tf.get_variable_scope().reuse)
	    with tf.variable_scope('root',reuse=True):
	        print(tf.get_variable_scope().reuse)
	        with tf.variable_scope('bar'):
	            print(tf.get_variable_scope().reuse)
	            with tf.variable_scope('bar1'):
	                print(tf.get_variable_scope().reuse)
	        print(tf.get_variable_scope().reuse)
	    print(tf.get_variable_scope().reuse)
	    
	#在命名空间内创建的变量的名称都会带上这个命名空间名做前缀
	v1 = tf.get_variable("v", [1])
	print(v1.name)
	
	with tf.variable_scope("foo",reuse=True):
	    v2 = tf.get_variable("v", [1])
	print(v2.name)
	
	with tf.variable_scope("foo"):
	    with tf.variable_scope("bar"):
	        v3 = tf.get_variable("v", [1])
	        print(v3.name)
	        
	v4 = tf.get_variable("v1", [1])
	print(v4.name)
	
	#通过变量的名称来获取变量
	with tf.variable_scope("",reuse=True):                                                                                                      
	    v5 = tf.get_variable("foo/bar/v", [1])
	    print(v5 == v3)
	    v6 = tf.get_variable("v1", [1])     
	    print(v6 == v4)
	
## 2、模型持久化代码实现
### 1、ckpt文件保存方法
1.保存模型:使用tf.train.Saver()，具体使用如下：  

	v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
	v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
	result = v1 + v2
	init_op = tf.global_variables_initializer()
	saver = tf.train.Saver()#获得API
	with tf.Session() as sess:
	    sess.run(init_op)
	    saver.save(sess, "Saved_model/model.ckpt")#保存
**注意：**以上代码将Tensorflow模型保存至ckpt文件中，虽然该方法只指定一个路径，但是却创建了3个文件，分别是model.ckpt.meta，保存了计算图的结构，第二个文件是model.ckpt，该文件保存了每个变量的取值，最后一个文件是checkpoint，保存了一个目录下所有模型文件列表  
2.加载模型：使用saver.restore(sess, "Saved_model/model.ckpt")，具体使用如下；

	saver=tf.train.Saver()                                                                                                                   
	with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print sess.run(result)  
3.直接加载持久化的图：tf.train.import_meta_graph("Saved_model/model.ckpt.meta")，具体使用如下：

	saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
	with tf.Session() as sess:
	    saver.restore(sess, "Saved_model/model.ckpt")
	    print sess.run(tf.get_default_graph().get_tensor_by_name("add:0")) 
3.变量重命名，具体使用如下：

	v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
	v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")
	saver = tf.train.Saver({"v1": v1, "v2": v2})
### 2、滑动平均类的保存
1.使用滑动平均：

	v = tf.Variable(0, dtype=tf.float32, name="v")
	for variables in tf.global_variables(): print variables.name
	ema = tf.train.ExponentialMovingAverage(0.99)
	maintain_averages_op = ema.apply(tf.global_variables())
	for variables in tf.global_variables(): print variables.name
2.保存滑动平均模型

	saver = tf.train.Saver()
	with tf.Session() as sess:
	    init_op = tf.global_variables_initializer()
	    sess.run(init_op)
	    sess.run(tf.assign(v, 10))
	    sess.run(maintain_averages_op)
	    # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
	    saver.save(sess, "Saved_model/model2.ckpt")
	    print sess.run([v, ema.average(v)])
3.加载滑动平均模型

	v = tf.Variable(0, dtype=tf.float32, name="v")
	# 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
	saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
	with tf.Session() as sess:
	    saver.restore(sess, "Saved_model/model2.ckpt")
	    print sess.run(v)
4.variables_to_restore函数的使用样例 (unsaved changes)

	import tensorflow as tf
	v = tf.Variable(0, dtype=tf.float32, name="v")
	ema = tf.train.ExponentialMovingAverage(0.99)
	print ema.variables_to_restore()
	saver = tf.train.Saver(ema.variables_to_restore())
	with tf.Session() as sess:
	    saver.restore(sess, "Saved_model/model2.ckpt")
	    print sess.run(v)

### 3、pb文件保存方法 (unsaved changes)
&nbsp;&nbsp;当不需要存储全部信息时，可使用如下方法将计算图中的变量及其取值通过常量的方式保存。
1.保存：

	import tensorflow as tf
	from tensorflow.python.framework import graph_util
	
	v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
	v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
	result = v1 + v2
	
	init_op = tf.global_variables_initializer()
	with tf.Session() as sess:
	    sess.run(init_op)
	    graph_def = tf.get_default_graph().as_graph_def()#导出当前图的GraphDef部分即可完成从输入层到输出层的计算过程
	    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
	    with tf.gfile.GFile("Saved_model/combined_model.pb", "wb") as f:
	           f.write(output_graph_def.SerializeToString())
2.加载：

	from tensorflow.python.platform import gfile
	with tf.Session() as sess:
	    model_filename = "Saved_model/combined_model.pb"
	   	#读取保存的模型文件
	    with gfile.FastGFile(model_filename, 'rb') as f:
	        graph_def = tf.GraphDef()
	        graph_def.ParseFromString(f.read())
		#将保存的图读取到当前图中，return_elements表示返回的张量的名称，张量名称为节点名称后面加上（:0）
	    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
	    print sess.run(result)
## 3、持久化数据格式
&nbsp;&nbsp;在文件model.ckpt.meta中存储了元图的数据，但该文件是二进制的，无法直接查看，TensorFlow有将其以json格式导出的方法：
	
	v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
	v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
	result = v1 + v2
	saver = tf.train.Saver()
	saver.export_meta_graph("Saved_model/model.ckpt.meta.json",as_text=True)#导出元图并以json格式输出
&nbsp;&nbsp;查看保存的变量信息：

	reader=tf.train.NewCheckpointReader('Saved_model/model.ckpt')#获取所有变量列表
	all_variable=reader.get_variable_to_shape_map()
	for variable_name in all_variable:
	    print(variable_name,all_variable[variable_name])
	print(1,reader.get_tensor('v1'))









































