# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:17:33 2017

@author: ZhifengFang
"""
#解决乱码问题
#位置是C:\Python27\lib\site-packages\matplotlib\mpl-data\matplotlibrc
#打开matplotlibrc文件进行编辑，找到#font.family : sans-serif更改为： font.family : SimHei
#不适合在较大的应用程序中使用
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

x=np.linspace(0,10,1000)
y=np.sin(x)
z=np.cos(x)
matplotlib.rcParams["savefig.dpi"]
plt.figure(figsize=(8,4))#设置大小，英寸、整个图表都是一个Figure对象
plt.plot(x,y,label='$sin(x)$',color='red',linewidth=2)#在Axes（子图）对象上绘图
plt.plot(x,z,'b--',label="$cons(x^2)$")#b代表颜色，--表示虚线

plt.xlabel("Time(S)")#x轴的标题
plt.ylabel("Volt")#y轴的标题
plt.title("example")#整个图的标题
plt.ylim(-2,2)#y轴的限制
plt.xlim(0,10)
plt.legend()#画图示
plt.savefig('test.png',dpi=120)#保存图片，大小是8*120px 4*120px
plt.show()#整张图显示


x=np.arange(0,5,0.1)
line,=plt.plot(x,x*x,'b--')#画图并得到第一个元素
print("-------------获得对象的属性-----------------")
print(line.get_linewidth())#线宽度
print(plt.getp(line,'color'))#颜色
print(plt.getp(line))#全部属性
f=plt.gcf()#获取当前的绘图对象Get Qurait Figure
print(plt.getp(f))
print(plt.getp(f,'axes'))#图表中的一个子图
print(plt.gca())#当前子图Get Current Axes
line.set_antialiased(False)#关闭反锯齿效果
lines=plt.plot(x,np.sin(x),x,np.cos(x))#画两个图并获得两个对象

plt.setp(lines,color='r',linewidth=2.0)#两个对象一起设置属性
plt.legend()
plt.show()

 
for idx ,color in enumerate('rgbyck'):
    plt.subplot(320+idx+1,axisbg=color)

plt.show()


plt.subplot(221)#绘制多子图，subplot(numRows, numCols, plotNum)，总行数，总列数，编号
plt.subplot(222)
ax1=plt.subplot(212)#返回的是子图
plt.show()


print("-----------------------配置文件------------------------------")
print(matplotlib.get_configdir())#获取用户配置路径
print(matplotlib.matplotlib_fname())#获取目录使用的配置文件路径

import os
print(os.getcwd())#将matplotlibrc复制一份到脚本的当前目录下
print(matplotlib.rcParams)#配置文件的读入
matplotlib.rcParams["lines.marker"]='o'#修改配置文件信息
matplotlib.rc('lines',marker='x',linewidth=2,color='red')
import pylab
pylab.plot([1,2,3])
matplotlib.rcdefaults()#恢复到默认值
matplotlib.rcParams.update(matplotlib.rc_params())#手工修改了配置文件后看，载入最新配置

pylab.show()
print("----------------------------Artist------------------------")

fig=plt.figure()#获得Figure对象
fig.show()
fig.patch.set_color('g')#设置某个属性
fig.set(alpha=0.5,zorder=2)#设置多个属性
fig.canvas.draw()#更新显示
print(plt.getp(fig.patch))#获得属性名、值，这样可以查看
ax=fig.add_axes([0.15,0.1,0.7,0.3])#添加一个子图，并设置[left, bottom, width, height]
ax.set_xlabel('time')
line,=ax.plot([1,2,3],[1,2,1])#123代表x轴的点，121代表y轴对应的点
print(ax.xaxis.label.get_text())
print("----------------------------Figure------------------------")


























