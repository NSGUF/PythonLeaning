# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:34:08 2017

@author: ZhifengFang
"""
#==============================================================================
# print("我是方智峰","你呢？",1,2,True)#逗号会换成空格，然后拼接
# name=input("你的名字：")#输入字符串
#==============================================================================

#==============================================================================
# 数值：整数、浮点数、字符串、空值、布尔
# 布尔运算：and or not
# str运算：'1'.replace('1','2')
# isinstance(x, (int, float))检查x是否属于int或float
#==============================================================================

#==============================================================================
# print(ord('a'))#编码
# print(chr(97))
#==============================================================================


#==============================================================================
# #list 其中元素可以不同
# names=['Bob','Mary','Jack']
# print(len(names))
# print(names[0])
# print(names[-1])
# names.append('Dav')#追加
# names.count(1)# 统计a中元素1出现的次数
# names.extend([1,2])# 将列表[1,2]的内容追加到a的末尾中
# names.index('Bob')# 第一个Bob的索引位置
# names.insert(1,'Kery')#插入，第一个参数是位置
# names.pop(3)#删除指定位置
# names[1]='Arry'#替换
# names.sort()#排序
# 如果想复制names，不能使用name=names，因为name只是引用了names，正确方法是：
# name=names[:]
#==============================================================================


#==============================================================================
# #tuple 元组，和list相似（中括号换成小括号，其他方法一样可以用），但是初始化后不能改
# #因为在数学公式中也有小括号，所以有一个元素是必须加一个逗号，
# t=(1,)
# #tuple的不变表示指向不变，比如元组中有一个list，list的指向不变，但是list中的值可以变
# t=(1,2,[1,2])
# t[2][0]=5
#==============================================================================

#==============================================================================
# #条件判断
# cl=5
# if cl<=6:
#     print('小学')
# elif cl<=9:
#     print('初中')
# elif cl<=12:
#     print('高中')
# else:
#     print('大学或没读')
# #循环
# names=['Bob','Mary','Jack']
# for name in names:
#     print(name)
# n=0
# while n<len(names):
#     print(names[n])
#     n=n+1
#==============================================================================

#==============================================================================
# #dict 字典，键值对形式，查找速度快，key不可变，所以key不能为list
# d={'1':1,'2':2}
# d=dict([['1',1])
# print(d['1'])
# d['1']=11#改变值
# if '1' in d:#避免key不存在报错
#     print(d['1'])
# print(d.get('1'))#如果key不存在，返回None
# d.pop('1')#删除该值，value也会被删除掉
# for k,v in d.items:
#     print(k,v)
#==============================================================================

#==============================================================================
# dict内部存放的顺序和key放入的顺序是没有关系的。
# 和list比较，dict有以下几个特点：
# 查找和插入的速度极快，不会随着key的增加而变慢；
# 需要占用大量的内存，内存浪费多。
# 而list相反：
# 所以，dict是用空间来换取时间的一种方法。
#==============================================================================

#==============================================================================
# #set key的集合， 不能重复，
# s={1,2,3,4,2,1}
# s1=set([1,2,3,4,1,23,3,4])#重复元素自动过滤
# s1.add(20)
# s1.remove(20)
# s2=set([5,4,3])
# #set可以看成数学意义上的无序和无重复元素的集合，因此，两个set可以做数学意义上的交集、并集等操作：
# print(s1&s2)# 交集
# print(s1|s2)# 并集
# print(s1-s2)# 差集
# print(s1^s2)# 对称差集
#==============================================================================

#==============================================================================
# #python内置函数
# abs(100)
# max(1,2,3,5)
# min(1,2,3,4)
# print(round(3.4543,2))# 表示四舍五入，第二个参数表示几个小数点
# int('123')
# cmp(a,b)# 比较两个列表/元组的元素
# float('123.43')
# str(1.24)
# bool(1)
#==============================================================================

#==============================================================================
# #函数
# def nop():
#     pass
#==============================================================================

#==============================================================================
# #参数
# *args是可变参数，args接收的是一个tuple；
# **kw是关键字参数，kw接收的是一个dict。
#==============================================================================

#==============================================================================
# #切片 list、tuple、str可用
# names=['Bob','Mary','Jack']
# print(names[0:2])
# print(names[:2])
# print(names[-2:])#
# print(names[:3:2])#每两个取一个
#==============================================================================

#==============================================================================
# #迭代 
# from collections import Iterable
# print(isinstance('abc',Iterable))
# #内置的enumerate函数可以把一个list变成索引-元素对
# for i,value in enumerate(['1','2','3','4']):
#     print(i,value)
#==============================================================================

#==============================================================================
# #列表生成式
# print(list(range(1,33)))
# l=[x*x for x in range(1,4)]
# print(l)
# l=[x*x for x in range(1,11) if x%2==0]
# print(l)
# l=[m+n for m in 'ABC' for n in 'XYZ']
# print(l)
# 
#==============================================================================

#==============================================================================
# #生成器 generator  generator保存的是算法
# g=(x*x for x in range(1,4))#将中括号换成小括号就是生成器
# print(next(g))#打印下一个值，到最后一个会报错，所以用for循环
# #将print换成yield ，函数变成了generator
#==============================================================================

#==============================================================================
# #map map(fun,Iterable) 将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回
# def f(x):
#     return x*x
# 
# print(list(map(f,[1,2,3,4])))
# #reduce reduce(fun,Iterable) 把结果继续和序列的下一个元素做累积计算
# from functools import reduce
# def add(x,y):
#     return x+y
# print(reduce(add,[1,2,3,4]))
# #filter filter(fun,Iterable) 把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。
# def is_odd(x):
#     return x%2==1
# 
# print(list(filter(is_odd,[1,2,3,4,5])))
#==============================================================================

#=============排序问题=================================================================
# def by_name(t):
#     return t[0]
# L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
# print(sorted(L,key=by_name))
# print(sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True))#忽略大小写
# 
#==============================================================================

#==============================================================================
# #匿名函数
# print(list(map(lambda x:x*x,[1,2,3])))
#==============================================================================

#==============================================================================
# #闭包
# def count():
#     fs = []
#     def f(x):
#         return lambda: x*x#通过匿名函数
#     for i in range(1, 4):
#         fs.append(f(i))
#     return fs
# 
# f1,f2,f3=count()
# print(f1(),f2(),f3())
#==============================================================================

#==============================================================================
# #装饰器  运行看看
# import functools
# 
# def logger(text):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kw):
#             print('%s %s():' % (text, func.__name__))
#             return func(*args, **kw)
#         return wrapper
#     return decorator
# 
# @logger('DEBUG')
# def today():
#     print('2015-3-25')
# 
# today()
#==============================================================================

#==============================================================================
# #偏函数 可以降低函数调用的难度
# import functools
# max2=functools.partial(max,10)
# print(max2(1,2,3,4))#自动把10放在比较数值中
#==============================================================================

#==============================================================================
# #一个.py文件称之为一个模块
#==============================================================================

#==============================================================================
# #类和实例 #变量名类似__xxx__的，也就是以双下划线开头，并且以双下划线结尾的，是特殊变量，特殊变量是可以直接访问的，不是private变量，
# class Student(object):
#     def __init__(self,name,score):#_init__方法的第一个参数永远是self，表示创建的实例本身
#         self.__name=name#表示私有
#         self.__score=score#_name，这样的实例变量外部是可以访问的，意思就是，“虽然我可以被访问，但是，请把我视为私有变量，不要随意访问”。
#     def print_score(self):
#         print(self.__name,self.__score)
#     def get_name(self):
#         return self.__name
#     def set_name(self,name):
#         self.__name=name
#     def set_score(self,score):
#         self.__score=score
#     def get_score(self):
#         return self.__score
# s=Student('Bart',14)
# #__name不能访问是因为将其改成了_St05udent__name
# print(s._Student__name)
# s.__name='123'#内部的__name变量已经被Python解释器自动改成了_Student__name，而外部代码给bart新增了一个__name变量。
# print(s.get_name())#没有被改变
# s.print_score()
# 
# #继承和多态  
# class ChuStudent(Student):#初中生继承了学生
#     def set_score(self,score):
#         if score<=100:
#             self.__score=score
#         else:
#             print('得分有误')
# chu=ChuStudent('123',40)
# chu.set_score(143)
# chu.print_score()
# 
# #获取对象类型
# print(type(123))
# import types#判断一个对象是否是函数
# print(type(lambda x:x*x)==types.LambdaType)
# print(isinstance('a', str))
# print(dir('123'))#获取全部属性和方法
# print(hasattr(chu,'name'))#是否有name属性
# setattr(chu,'age',10)#设置属性
# print(getattr(chu,'age'))#获取属性
# print(chu.age)#获取属性 如果不存在会抛出AttributeError的错误
# print(getattr(chu,'z',404))#最后赋予默认值
# #绑定方法
# def set_age(self,age):
#     self.age=age
# from types import MethodType
# s.set_age=MethodType(set_age,s)
# s.set_age(120)
# print(s.age)
# #若给所有实例绑定方法
# ChuStudent.set_age=set_age
# 
# #限制实例的属性
# class XiaoStudnet(Student):
#     __slots__=('name','score','age')# 用tuple定义允许绑定的属性名称
# 
#==============================================================================

# =============================================================================
# #@property
# class Student(object):#birth是可读写属性，而age就是一个只读属性，
#     @property
#     def birth(self):
#         return self._birth
# 
#     @birth.setter
#     def birth(self, value):
#         self._birth = value
# 
#     @property
#     def age(self):
#         return 2015 - self._birth
# 
# #多重继承
# class Object1(Object2,Object3....):
#     pass
# =============================================================================

# =============================================================================
# #位实例添加一个打印显示
# class Student(object):
#     def __init__(self, name):
#         self.name = name
#     def __str__(self):
#         return 'Student object (name=%s)' % self.name
#     __repr__ = __str__
# 
# print(Student("123"))
# =============================================================================

# =============================================================================
# #如果一个类想被用于for ... in循环，类似list或tuple那样，就必须实现一个__iter__()方法，该方法返回一个迭代对象
# class Fib(object):
#     def __init__(self):
#         self.a, self.b = 0, 1 # 初始化两个计数器a，b
# 
#     def __iter__(self):
#         return self # 实例本身就是迭代对象，故返回自己
#     
#     def __getitem__(self, n):#直接f[]显示
#         if isinstance(n, int): # n是索引
#             a, b = 1, 1
#             for x in range(n):
#                 a, b = b, a + b
#             return a
#         if isinstance(n, slice): # n是切片
#             start = n.start
#             stop = n.stop
#             if start is None:
#                 start = 0
#             a, b = 1, 1
#             L = []
#             for x in range(stop):
#                 if x >= start:
#                     L.append(a)
#                 a, b = b, a + b
#             return L
#     def __getattr__(self, attr):#防止不存在的属性，报错
#         if attr=='score':
#             return 99
#     def __call__(self):#可以直接对实例进行调用 s()
#         print('My name is %s.' % self.name)
#     def __next__(self):
#         self.a, self.b = self.b, self.a + self.b # 计算下一个值
#         if self.a > 100000: # 退出循环的条件
#             raise StopIteration()
#         return self.a # 返回下一个值
# =============================================================================

# =============================================================================
# #枚举类
# from enum import Enum
# Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
# for name, member in Month.__members__.items():
#     print(name, '=>', member, ',', member.value)
# 
# from enum import Enum, unique
# 
# @unique#@unique装饰器可以帮助我们检查保证没有重复值。
# class Weekday(Enum):
#     Sun = 0 # Sun的value被设定为0
#     Mon = 1
#     Tue = 2
#     Wed = 3
#     Thu = 4
#     Fri = 5
#     Sat = 6
# =============================================================================

# =============================================================================
# #错误处理
# try:
#     r=10/0
# except Exception as e:
#     print(e)
# finally:
#     print('finally...')
#     
# print('end')
# =============================================================================

# =============================================================================
# #读写文件
# try:
#     f=open('test.txt','r')
#     print(f.read())
# finally:
#     if f:
#         f.close()
# 
# with open('test.txt','r') as f:
#     print(f.read())
#     #防止文件过大，一行一行读
#     for line in f.readline():
#         print(line.strip())
#     
# with open('test.txt','w') as f:
#     f.write('Hello,world!')
# =============================================================================

#==============================================================================
# #StringIO 内存中读写str
# from io import StringIO
# f=StringIO()
# print(f.write('hello'))
# s=f.getvalue()
#==============================================================================

#==============================================================================
# from io import StringIO
# f=StringIO('Hello!')
# while True:
#     s=f.readline()
#     if s=='':
#         break
#     print(s.strip())
#==============================================================================

#==============================================================================
# from io import BytesIO
# f=BytesIO()
# print(f.write("中文".encode('utf-8')))
# print(f.getvalue())
#==============================================================================

#==============================================================================
# from io import BytesIO
# f=BytesIO(b'\xe4\xb8\xad\xe6\x96\x87')
# print(f.read())
#==============================================================================

#==============================================================================
# #操作文件和目录
# import os
# print(os.name)#posix是linux系统，nt是windows系统
# print(os.environ)#环境变量
# print(os.environ.get('PATH'))#某个环境变量
# print(os.path.abspath('.'))#查看当前目录的绝对路径
# print(os.path.join('C:\\Users\\Administrator\\Desktop\\PythonLeaning','testdir'))#将两个目录拼起来
# print(os.mkdir('C:\\Users\\Administrator\\Desktop\\PythonLeaning\\testdir'))#创建目录
# print(os.rmdir('C:\\Users\\Administrator\\Desktop\\PythonLeaning\\testdir'))
# print(os.path.split('C:\\Users\\Administrator\\Desktop\\PythonLeaning'))#拆分路径，最后一个拆出来
# print(os.path.splitext('C:\\Users\\Administrator\\Desktop\\PythonLeaning\\test.txt'))#获得文件扩展名
# #这些合并、拆分路径的函数并不要求目录和文件要真实存在，它们只对字符串进行操作。
# print(os.rename('test.txt','test.py'))#重命名
# print(os.remove('test.py'))#删除文件
# print([x for x in os.listdir('.') if os.path.isdir(x)])#累出当前目录下所有目录
# print([x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.py'])#列出所有.py的文件
#==============================================================================

#==============================================================================
# #序列化 ：把变量从内存中变成可存储或传输的过程称之为序列化
# import pickle
# d=dict(name='Bob',age=20,score=88)# 创建字典
# f=open('dump.txt','wb')
# pickle.dump(d,f)
# f.close()
# f=open('dump.txt','rb')
# d=pickle.load(f)
# f.close()
# print(d)
#==============================================================================

#==============================================================================
# #JSON 
# import json
# d=dict(name='Bob',age=20,score=99)# 创建字典
# d=dict.fromkeys(['Bob','Mary'],10)# 创建字典
# print(json.dumps(d))#返回的是标准的json格式
# json_str='{"name": "Bob", "age": 20, "score": 99}'
# print(json.loads(json_str))
# #实例序列化
# class Student(object):
#     def __init__(self,name,age,score):
#         self.name=name
#         self.age=age
#         self.score=score
#         
# def student2dit(std):
#     return {
#             'name':std.name,
#             'age':std.age,
#             'score':std.score
#             }
# s=Student('Bob',20,99)
# print(json.dumps(s,default=student2dit))
# print(json.dumps(s,default=lambda obj:obj.__dict__))#通常class的实例都有一个__dict__属性，它就是一个dict，用来存储实例变量。
# #反序列化
# def dict2student(d):
#     return Student(d['name'],d['age'],d['score'])
# print(json.loads(json_str,object_hook=dict2student))
#==============================================================================

#==============================================================================
# #线程 python的os模块封装了常见的系统调用，
# import os
# from multiprocessing import Process
#==============================================================================

#==============================================================================
# #子线程要执行的代码
# def run_proc(name):
#     print(name,os.getpid())
#     
# if __name__=='__main__':
#     print(os.getpid())
#     p=Process(run_proc('test'))
#     print('启动')
#     p.start()
#     p.join()#等待子进程结束后再继续往下运行，通常用于进程间的同步。
#     print('结束')
#==============================================================================

#==============================================================================
# #如果要启动大量的子进程，可以用进程池的方式批量创建子进程
# from multiprocessing import Pool
# import time,random
# def long_time_task(name):
#     print(name,os.getpid())
#     start=time.time()
#     time.sleep(random.random()*3)
#     end=time.time()
#     print(name,(end-start))
# 
# if __name__=='__main__':
#     print('等待完成')
#     print(os.getpid())
#     p=Pool(4)
#     for i in range(5):
#         p.apply_async(long_time_task(i))
#     p.close()
#     p.join()
#     print('完成')
#==============================================================================

#==============================================================================
# from multiprocessing import Process, Queue
# import os, time, random
# 
# # 写数据进程执行的代码:
# def write(q):
#     print('Process to write: %s' % os.getpid())
#     for value in ['A', 'B', 'C']:
#         print('Put %s to queue...' % value)
#         q.put(value)
#         time.sleep(random.random())
# 
# # 读数据进程执行的代码:
# def read(q):
#     print('Process to read: %s' % os.getpid())
#     while True:
#         value = q.get(True)
#         print('Get %s from queue.' % value)
# 
# if __name__=='__main__':
#     # 父进程创建Queue，并传给各个子进程：
#     q = Queue()
#     pw = Process(target=write(q,))
#     pr = Process(target=read(q,))
#     # 启动子进程pw，写入:
#     pw.start()
#     # 启动子进程pr，读取:
#     pr.start()
#     # 等待pw结束:
#     pw.join()
#     # pr进程里是死循环，无法等待其结束，只能强行终止:
#     pr.terminate()
#==============================================================================

#==============================================================================
# #全局变量local_school就是一个ThreadLocal对象，每个Thread对它都可以读写student属性，但互不影响。你可以把local_school看成全局变量，但每个属性如local_school.student都是线程的局部变量，可以任意读写而互不干扰，也不用管理锁的问题，ThreadLocal内部会处理。
# #可以理解为全局变量local_school是一个dict，不但可以用local_school.student，还可以绑定其他变量，如local_school.teacher等等。
# #ThreadLocal最常用的地方就是为每个线程绑定一个数据库连接，HTTP请求，用户身份信息等，这样一个线程的所有调用到的处理函数都可以非常方便地访问这些资源。
# import threading
# 
# # 创建全局ThreadLocal对象:
# local_school = threading.local()
# 
# def process_student():
#     # 获取当前线程关联的student:
#     std = local_school.student
#     print('Hello, %s (in %s)' % (std, threading.current_thread().name))
# 
# def process_thread(name):
#     # 绑定ThreadLocal的student:
#     local_school.student = name
#     process_student()
# 
# t1 = threading.Thread(target= process_thread, args=('Alice',), name='Thread-A')
# t2 = threading.Thread(target= process_thread, args=('Bob',), name='Thread-B')
# t1.start()
# t2.start()
# t1.join()
# t2.join()
#==============================================================================




















































