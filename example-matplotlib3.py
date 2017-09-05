# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:30:39 2017

@author: ZhifengFang
"""

import numpy as np
import matplotlib.pyplot as plt

map = {
    'Female': (0.5923, '#7199cf'),
    'Male': (0.3765, '#4fc4aa'),
    'other': (0.0312, '#e1a7a2')
}

fig = plt.figure(figsize=(8,16))
ax = fig.add_subplot(211)#添加一个子图
ax.set_title('Gender of friends')

xticks = np.arange(3)+0.15# 生成x轴每个元素的位置
bar_width = 0.5# 定义柱状图每个柱的宽度

names = map.keys()#获得x轴的值
values = [x[0] for x in map.values()]# y轴的值
colors = [x[1] for x in map.values()]# 对应颜色

bars = ax.bar(xticks, values, width=bar_width, edgecolor='none')# 画柱状图，横轴是x的位置，纵轴是y，定义柱的宽度，同时设置柱的边缘为透明
ax.set_ylabel('Proprotion')# 设置标题
ax.set_xlabel('Gender')
ax.grid()#打开网格
ax.set_xticks(xticks)# x轴每个标签的具体位置
ax.set_xticklabels(names)# 设置每个标签的名字
ax.set_xlim([bar_width/2-0.5, 3-bar_width/2])# 设置x轴的范围
ax.set_ylim([0, 1])# 设置y轴的范围

for bar, color in zip(bars, colors):
    bar.set_color(color)# 给每个bar分配指定的颜色
    height=bar.get_height()+0.01#获得高度并且让字居上一点
    plt.text(bar.get_x()+bar.get_width()/4.,height,'%.2f%%' %float(height*100))#写值

plt.show()


plt.figure(figsize=(8,8))
plt.title("123")
labels = ['{}\n{}'.format(name, value) for name, value in zip(names, values)]
plt.pie(values, labels=labels, colors=colors)
plt.show()
