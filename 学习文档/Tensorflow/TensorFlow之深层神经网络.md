# 第四章 深层神经网络
&nbsp;&nbsp;深度学习所示深层神经网络的代名词，重要特性：多层、非线性。  
&nbsp;&nbsp;若只通过线性变换，任意层的神经网络模型与单层神经网络模型的表达能力没有任何区别，这是线性模型的局限性。对于线性可分的问题中，线性模型可解决，但在现实生活中，绝大部分的问题都是无法线性分割的。  
&nbsp;&nbsp;感知机：单层神经网络。不能处理异或问题。
## 1、激活函数
&nbsp;&nbsp;将每一个神经元（神经网络的节点）的输出通过一个非线性函数便可使得整个神经网络的模型非线性化，这个非线性函数就是激活函数。  
&nbsp;&nbsp;常用非线性激活函数：tf.nn.relu、tf.sigmoid、tf.tanh；使用方法，例：  

	tf.nn.relu(tf.matmul(x,w1)+biases1)  

&nbsp;&nbsp;**偏置项：**可理解为数学中y=ax+b中的b，如果在分类的情况下，两点刚好在经过原点的直线上，如果没有偏置项b的话，无法划分直线将两个点分开。
## 2、损失函数
&nbsp;&nbsp;概念：用来评价模型的预测值Y^=f(X)与真实值Y的不一致程度，它是一个非负实值函数。通常使用L(Y,f(x))来表示，损失函数越小，模型的性能就越好。  

* 经典损失函数  

1.分类问题：将不同的样本分到事先定义好的类别中。常用交叉熵计算；计算方法：  

	cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
	#y_正确结果，y测试结果；y_*tf.log(tf.clip_by_value(y,1e-10,1.0))得到n*m的二维矩阵，
	#n为样例的数量，m为分类的类别数量。由于交叉熵一般与softmax回归一起使用，所以tf对其进行统一封装，  
	cross_entropy=tf.nn.softmax_cross_entropy_with_logits(y,y_)
	 

&nbsp;&nbsp;2.回归问题：对具体数值的预测。常用均方差；计算方法：  
	
	mse=tf.reduce_mean(tf.square(y_-y)#y_标准答案，y输出答案
* 自定义损失函数  
&nbsp;&nbsp;除了分类和回归还有其他问题，所以可以根据具体情况，具体写对应的损失函数。
## 3、神经网络优化算法  
&nbsp;&nbsp;目的：使损失函数尽可能的小；  

* 反向传播算法（backpropagtion）：给出一个高效的方式在所有参数上使用梯度下降算法；是训练神经网络的核心算法，可以根据定义好的损失函数优化神经网络中参数的取值；
* 梯度下降算法（gradient decent):优化单个参数的取值  
&nbsp;&nbsp;当要求损失函数的值为最小值时，可以根据其偏导来判断其下降方向，梯度便是偏导值；当损失函数未到最小值时，（损失函数-梯度）可变的更小，使用反向传播算法，可一直将损失函数减去其梯度，知道获得最小的损失函数；其中由于梯度可能过大，导致错过最小值，所以可在梯度的值乘以学习率。即：下一个损失函数=损失函数-梯度*学习率（其中梯度便是损失函数的偏导）下面给出一个损失函数为y=x^2的例子：  

		TRAINING_STEPS=5
		x=tf.Variable(tf.constant(5,dtype=tf.float32),name='x')
		y=tf.square(x)
		train_op=tf.train.GradientDescentOptimizer(0.3).minimize(y)#实现梯度下降算法的优化器
		with tf.Session() as sess:
		    sess.run(tf.global_variables_initializer())
		    for i in range(TRAINING_STEPS):
		        sess.run(train_op)
		        x_value=sess.run(x)
		        print(i+1, x_value)  
![](https://i.imgur.com/c8FDpPz.png)  
**注意：**梯度下降算法并不可保证优化的函数能得到全局最优解，只有损失函数为凸函数时才可保证；样例如下图
![](https://i.imgur.com/351p4k1.png)  
* 随机梯度下降算法（stochastic gradient decent）：梯度下降算法中，由于计算的是全部训练数据上最小化最优，所以损失函数是在所有训练数据上的损失和，这样每轮迭代导致时间过长；由此问题引出随机梯度下降算法，随机优化某一条训练数据上的损失函数。该方法虽然达到时间缩短，但是会导致函数无法得到全局最优。
* 实际采用方法：梯度与随机梯度的折中--每次计算一小部分训练数据的损失函数（batch），该方法可大大减少收敛所需的迭代次数，缩短时间，同时可使收敛到的结果最优。神经网络的训练大都遵循以下过程：  

	   	batch_size=n
		#读取一小部分数据作为当前训练集
		x=tf.placeholder(tf.float32,shape=[None,2],name='x-input')
		x=tf.placeholder(tf.float32,shape=[None,1],name='y-input')
		#定义神经网络结构和优化算法
		loss=...
		train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
		
		with tf.Session() as sess:
		    #参数初始化
		    ...
		    #迭代更新参数
		    for i in range(STEPS):
		        #将所有随机数据重新打乱之后再选取将会获得更好的优化效果
		        X=..
		        Y=...
		        sess.run(train_step,feed_dict={x:X,y:Y})   
  

## 4、神经网络的进一步优化
* 学习率的设置  
&nbsp;&nbsp;在梯度下降算法中，如果学习率取值过大，可能会出现数值两边震荡的情况，永远都到达不了极值，而若学习率取值过小，则会大大降低优化的速度，TensorFlow提供了学习率设置的方法--指数衰减法，更改上面的梯度例子如下：  

		TRAINING_STEPS=100
		global_step=tf.Variable(1)#初始化为0
		#def exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
		#                      staircase=False, name=None):分别为：初始学习率，经过decay_steps轮计算，学习率乘以decay_rate，staircase为T，学习率阶梯状衰减，否则，连续衰减
		LEA_RATE=tf.train.exponential_decay(0.1,global_step,1,0.96,staircase=False)#获得学习率，指数衰减法，开始较大的衰减率，后来指数变小
		x=tf.Variable(tf.constant(5,dtype=tf.float32),name='x')
		y=tf.square(x)
		train_op=tf.train.GradientDescentOptimizer(LEA_RATE).minimize(y,global_step=global_step)#实现梯度下降算法的优化器，global_step会自动加1，可当计数器用
		with tf.Session() as sess:
		    sess.run(tf.global_variables_initializer())
		    for i in range(TRAINING_STEPS):
		        sess.run(train_op)
		
		        LEA_RATE_value=sess.run(LEA_RATE)
		        x_value=sess.run(x)
		        print(i+1, i+1, x_value, LEA_RATE_value,sess.run(global_step))

* 过拟合问题  
&nbsp;&nbsp;概念：当一个模型过为复杂之后，它会很好的‘记忆’每一个训练数据中随机随机噪音的部分而忘记了要去‘学习’训练数据中通用的趋势。意思就是过拟合数据中的随机噪音虽然可得到极小的损失函数，但是对位置数据可能无法做出可靠的判读。
&nbsp;&nbsp;解决方法：正则化--在损失函数中添加刻画模型复杂程度的指标。使用方法：  
    
		#lambda正则化项的权重，w需要计算正则化损失的参数，边上的权重
		loss=之前的损失函数+tf.contrib.layers.l2_regularizer(lambda)(w)

		print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(w)))#w中每个元素的绝对值之和乘0.5，所以值为5
		print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(w)))#w中每个元素的平方和除以2再乘0.5，结果为7.5
		print(sess.run(tf.contrib.layers.l1_l2_regularizer(0.5)(w)))#w中的每个元素之和乘0.5+w中每个元素的平方和乘（1-0.5）
&nbsp;&nbsp;当神经网络的参数增多时，损失函数也将会增多，若将其写在一个定义中，可读性会很差，容易出错，所以在TensorFlow中可以用一下方法：  

		#losses是集合的名字1，mse_loss是加入集合的内容
		tf.add_to_collection('losses', mse_loss)
		#get_collection返回的是losses集合，add_n是将集合的中值求和
		loss = tf.add_n(tf.get_collection('losses'))

## 5、滑动平均模型
&nbsp;&nbsp;作用：在使用随机梯度下降算法训练神经网络时，可只用该方法在一定程度上进行优化。使用方法如下：

	v1=tf.Variable(0,dtype=tf.float32)
	step=tf.Variable(0,trainable=False)
	#定义一个滑动平均模型的类，初始化衰减率为0.99，控制衰减率的变量step
	ema=tf.train.ExponentialMovingAverage(0.99,step)
	#定义一个更新变量滑动平均的操作，并给定一个列表，每次执行这个操作时，列表更新
	maintain_ave_op=ema.apply([v1])
	
	with tf.Session() as sess:
	    init_op=tf.global_variables_initializer()
	    sess.run(init_op)
		#上面的V1*衰减率+（1-衰减率）*更新变量v1，其中衰减率=min{初始化衰减率,（1+step）/（10+step）}
	    print(sess.run([v1,ema.average(v1)]))#结果：[0,0] 
	    
	    sess.run(tf.assign(v1,5))
	    sess.run(maintain_ave_op)
	    print(sess.run([v1,ema.average(v1)]))#0*min{0.99,(（1+step）/(step+10))}+(1-min{0.99,(（1+step）/(step+10))})*5
	    
	    sess.run(tf.assign(step,10000))
	    sess.run(tf.assign(v1,10))
	    sess.run(maintain_ave_op)
	    print(sess.run([v1,ema.average(v1)]))
	    
	    sess.run(maintain_ave_op)
	    print(sess.run([v1,ema.average(v1)]))


## 6、常用方法
* tf.reduce_sum() 和
* tf.greater(v1,v2)#v1和v2的每个对应元素比较，若v1大于v2，则为True否则为False 
* tf.where(tf.greater(v1,v2),v1,v2)#根据第一个参数的T和F来选择v1和v2，若为False则元素选V2中的值，整个函数相当于选取v1和v2中较大的数组成一个举证