# TensorBoard可视化
>TensorBoard是Tensorflow的可视化工具。可在程序运行过程中获取最新状态。下面来一个日志输出功能：
	
	import tensorflow as tf

	input1=tf.constant([1.0,2.0,3.0],name='input1')
	input2=tf.Variable(tf.random_uniform([3]),name='input2')
	
	output=tf.add_n([input1,input2],name='add')
	
	#生成写日志的writer
	writer=tf.summary.FileWriter('log',tf.get_default_graph())
	writer.close()

>然后在程序所在位置的命令行中输入 tensorboard --logdir=log，在到浏览器中打开localhost:6006即可。--port可以改变启动服务的端口。
## 命名空间
>为了更好地组织可视化效果图中的计算节点，TensorBoard支持通过TensorFlow命名空间来整理可视化效果图上的节点。对于命名空间的管理，Tensorflow给出了两个方法：tf.variable_scope，tf.name_scope，下面给出一个例子用于对比两者区别：

	with tf.variable_scope('b'):
	    print(tf.Variable([1],name='b').name)#b/b:0                                                                                                                                                                                                                                                                                                                                                                           
	    print(tf.get_variable('b', [1]).name)#b/b_1:0
	with tf.name_scope('a'):
	    print(tf.Variable([1]).name)#a/Variable:0
	    print(tf.get_variable('b', [1]).name)#b_1:0
	
	print(tf.Variable([1]).name)#Variable:0
	print(tf.get_variable('c', [1]).name)#c:0
>更改mnist_train.py的代码：

	def train(mnist):
	    #  输入数据的命名空间。
	    with tf.name_scope('input'):
	        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
	        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
	    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	    y = mnist_inference.inference(x, regularizer)
	    global_step = tf.Variable(0, trainable=False)
	    
	    # 处理滑动平均的命名空间。
	    with tf.name_scope("moving_average"):
	        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	        variables_averages_op = variable_averages.apply(tf.trainable_variables())
	   
	    # 计算损失函数的命名空间。
	    with tf.name_scope("loss_function"):
	        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	        cross_entropy_mean = tf.reduce_mean(cross_entropy)
	        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	    
	    # 定义学习率、优化方法及每一轮执行训练的操作的命名空间。
	    with tf.name_scope("train_step"):
	        learning_rate = tf.train.exponential_decay(
	            LEARNING_RATE_BASE,
	            global_step,
	            mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
	            staircase=True)
	
	        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	
	        with tf.control_dependencies([train_step, variables_averages_op]):
	            train_op = tf.no_op(name='train')
		#省略
	    writer = tf.summary.FileWriter("/log/modified_mnist_train.log", tf.get_default_graph())
	    writer.close()
## 节点信息
>TensorFlow不仅可以展示计算图的结构，还可将其节点基本信息以及运行时消耗的时间和空间显示出来。修改例子如下：

	def train(mnist):
		#省略
	    writer = tf.summary.FileWriter("C:\\Users\\Administrator\\Desktop\\PythonLeaning\\log", tf.get_default_graph())
	    # 训练模型。
	    with tf.Session() as sess:
	        tf.global_variables_initializer().run()
	        for i in range(TRAINING_STEPS):
	            xs, ys = mnist.train.next_batch(BATCH_SIZE)
	            if i % 1000 == 0:
	                # 配置运行时需要记录的信息。
	                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	                # 运行时记录运行信息的proto。
	                run_metadata = tf.RunMetadata()
	                _, loss_value, step = sess.run(
	                    [train_op, loss, global_step], feed_dict={x: xs, y_: ys},
	                    options=run_options, run_metadata=run_metadata)
	                writer.add_run_metadata(run_metadata,'step%03d' % i)#记录节点信息
	                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
	            else:
	                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
	    writer.close()











































