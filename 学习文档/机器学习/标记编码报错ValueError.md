>《Python机器学习经典实例》2.9小节中，想自己动手实践汽车特征评估质量，所以需要对数据进行预处理，其中代码有把字符串标记编码为对应的数字，如下代码
    
    input_data = ['vhigh', 'vhigh', '2', '2', 'small', 'low'] 
    input_data_encoded = [-1] * len(input_data)
    for i,item in enumerate(input_data):
        input_data_encoded[i] = int(label_encoder[i].transform(input_data[i]))

报错：  

	Traceback (most recent call last):
	  File "E:/17770426925/PythonLeaning/Machine-Learning/classifier/classifier.py", line 255, in <module>
	    input_data_encoded[i] = int(label_encoder[i].transform(input_data[i]))
	  File "D:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py", line 147, in transform
	    y = column_or_1d(y, warn=True)
	  File "D:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 562, in column_or_1d
	    raise ValueError("bad input shape {0}".format(shape))
	ValueError: bad input shape ()
所以由此看出，是label_encoder[i].transform(input_data[i])中input_data[i]输入的数值形式不对，需要将其改变成list，所以可对该代码进行改进：

    for i, item in enumerate(input_data):
        labels=[]
        labels.append(input_data[i])
        input_data_encoded[i] = int(label_encoder[i].transform(labels))