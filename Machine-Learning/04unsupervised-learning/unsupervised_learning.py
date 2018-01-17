# -*- coding: utf-8 -*-
"""
@Created on 2018/1/13 10:09

@author: ZhifengFang
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from sklearn.cluster import KMeans


def load_data(input_file):
    X = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data)
    return np.array(X)


# k_means例子
'''
# 1、获取数据并可视化
input_file = 'data_multivar.txt'
data = load_data(input_file)
x_min,x_max=min(data[:,0])-1,max(data[:,0])+1
y_min,y_max=min(data[:,1])-1,max(data[:,1])+1
plt.figure()
plt.scatter(data[:,0], data[:,1],
        facecolors='none', edgecolors='k')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.show()
# 2、获取k-means对象，并训练
kmeans=KMeans(n_clusters=4,init='k-means++',n_init=10)
kmeans.fit(data)
# 3、获取边界
step_size=0.01
x_values,y_values=np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))
predict_labels=kmeans.predict(np.c_[x_values.ravel(),y_values.ravel()])
predict_labels=predict_labels.reshape(x_values.shape)
# 4、画出边界
plt.figure()
plt.clf()
plt.imshow(predict_labels, interpolation='nearest',
           extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.scatter(data[:,0], data[:,1],
        facecolors='none', edgecolors='k')

centero=kmeans.cluster_centers_
plt.scatter(centero[:,0],centero[:,1],linewidths=5,facecolor='black')

plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.show()
'''

# 用矢量量化压缩图片
'''
import argparse

# 1、创建一个函数，用来解析输入参数，输入参数为图片和每个像素被压缩的比特数。
def build_arg_parser():
    parser = argparse.ArgumentParser(description='输入图片')
    parser.add_argument('--input-file', dest='input_file', required=True, help='输入图片')
    parser.add_argument('--num-bits', dest='num_bits', type=int, required=False, help='比特数')
    return parser


# 2、压缩输入图片
def compress_img(img, num_cluster):
    print(img)
    X = img.reshape(-1, 1)
    print(X)
    kmeans = KMeans(n_clusters=num_cluster, n_init=4, random_state=5)
    kmeans.fit(X)
    contrid = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    input_compress = np.choose(labels, contrid).reshape(img.shape)
    return input_compress

# 3、查看压缩算法对图片质量的影响
def plot_image(img,title):
    vmin=img.min()
    vmax=img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img,cmap=plt.cm.gray,vmin=vmin,vmax=vmax)

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    input_file=args.input_file
    num_bits=args.num_bits
    if not 1<=num_bits<=8:
        raise TypeError('比特数应该在1和8之间')
    num_clusters=np.power(2,num_bits)
    compression_rate=round(100*(8.0-args.num_bits)/8.0,2)
    input_image=misc.imread(input_file,True).astype(np.uint8)
    plot_image(input_image,'image')
    input_compress=compress_img(input_image,num_clusters)
    plot_image(input_compress,'rate='+str(compression_rate)+'%')
    plt.show()
'''

#