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

# 2、k_means例子
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

# 3、用矢量量化压缩图片
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

# 4、均值漂移聚类模型
'''
from sklearn.cluster import MeanShift, estimate_bandwidth

# 1、获取数据
X = load_data('data_multivar.txt')
# 2、通过指定输入参数创建均值漂移模型
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# 3、训练模型
meanshift.fit(X)
# 4、提取标记
labels = meanshift.labels_
# 5、获取集群中心点，并打印数量
centroids = meanshift.cluster_centers_
num_clusters = len(np.unique(labels))
print(labels)
print(centroids)
print(num_clusters, len(centroids))
# 6、可视化
markers = '.*xv'
plt.figure()
for i, marker in zip(range(len(markers)), markers):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='k')
    centroid = centroids[i]
    plt.plot(centroid[0], centroid[1], marker='o', markersize=15, markeredgecolor='k', markerfacecolor='k')
plt.title('显示')
plt.show()
'''

# 5、凝聚层次聚类
'''
def get_spiral(t,noise_amplitude=0.5):# 获取呈螺旋状的数据
    r=t
    x=r*np.cos(t)
    y=r*np.sin(t)
    return add_noise(x,y,noise_amplitude)
def get_rose(t,noise_amplitude=0.02):# 获取呈螺旋状的数据
    k=5
    r=np.cos(k*t)+0.25
    x=r*np.cos(t)
    y=r*np.sin(t)
    return add_noise(x,y,noise_amplitude)
def get_hypotrochoid(t, noise_amplitude=0):
    a, b, h = 10.0, 2.0, 4.0
    x = (a - b) * np.cos(t) + h * np.cos((a - b) / b * t)
    y = (a - b) * np.sin(t) - h * np.sin((a - b) / b * t)

    return add_noise(x, y, 0)
def add_noise(x,y,amplitude):# 添加噪音
    X=np.concatenate((x,y))
    X+=amplitude*np.random.randn(2,X.shape[1])
    return X.T
from sklearn.cluster import AgglomerativeClustering

def perform_clustering(X,connectivity,title,num_clusters=3,linkage='ward'):# 设置层次凝聚模型
    plt.figure()
    model=AgglomerativeClustering(linkage=linkage,n_clusters=num_clusters,connectivity=connectivity)
    model.fit(X)

    labels=model.labels_

    markers='.vx'

    for i,marker in zip(range(num_clusters),markers):
        plt.scatter(X[labels==i,0],X[labels==i,1],s=50,facecolor='none',marker=marker,color='k')
        plt.title(title)
from sklearn.neighbors import kneighbors_graph
if __name__=='__main__':
    # 生成样本数据
    n_samples=500
    np.random.seed(2)
    t=2.5*np.pi*(1+2*np.random.rand(1,n_samples))
    X=get_spiral(t)
    # X = get_rose(t)
    # X = get_hypotrochoid(t)
    connectivity=None
    perform_clustering(X,connectivity,'没有连接')


    connectivity=kneighbors_graph(X,10,include_self=False)
    perform_clustering(X,connectivity,'knei连接')# 可让连接在一起的数据组合在一起

    plt.show()
'''

# 6、评价聚类算法的聚类效果
'''
from sklearn import metrics
data=load_data('data_perf.txt')# 加载数据
scores=[]
range_values=np.arange(2,10)
for i in range_values:# 分别分i个集群
    kmean=KMeans(n_clusters=i,n_init=10,init='k-means++')
    kmean.fit(data)
    score=metrics.silhouette_score(data,kmean.labels_,metric='euclidean',sample_size=len(data))
    scores.append(score)

plt.figure()
plt.bar(range_values,scores,width=0.6,color='k',align='center')
plt.show()
plt.figure()
plt.scatter(data[:,0],data[:,1],color='k')
xmin,xmax=min(data[:,0])-1,max(data[:,0])+1
ymin,ymax=min(data[:,1])-1,max(data[:,1])+1
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()
'''

# 7、用DBSCAN算法自动估算集群数量
'''
# 1、获取数据
X=load_data('data_perf.txt')
# 2、初始化参数
eps_grid=np.linspace(0.3,1.2,num=10)
silhouette_scores=[]
eps_best=eps_grid[0]
silhouette_scores_max=-1
model_best=None
labels_best=None
# 3、以此执行所有参数
from sklearn.cluster import DBSCAN
from sklearn import metrics
for eps in eps_grid:
    model=DBSCAN(eps=eps,min_samples=5).fit(X)
    labels=model.labels_
    silhouette_score=round(metrics.silhouette_score(X,labels),4)
    silhouette_scores.append(silhouette_score)
    # 获取指标的最佳得分
    if silhouette_score>silhouette_scores_max:
        silhouette_scores_max=silhouette_score
        eps_best=eps
        model_best=model
        labels_best=labels

# 4、画出条形图
plt.figure()
plt.bar(eps_grid,silhouette_scores,width=0.05,color='k',align='center')
plt.show()
# 5、由于可能会有某些点还没有分配集群，所以这里删除未分配而获取集群的数量
offset=0
if -1 in labels:
    offset=1
num_clusters=len(set(labels))-offset
# 6、提取核心样本
model = model_best
labels = labels_best
mask_core=np.zeros(labels.shape,dtype=np.bool)# 初始化全部点的分配
mask_core[model.core_sample_indices_]=True# model.core_sample_indices_表示分配后的数据位置为True

# 7、数据可视化
from itertools import cycle
plt.figure()
labels_uniq=set(labels)
markers=cycle('vo^s<>')
for cur_label,marker in zip(labels_uniq,markers):
    if cur_label==-1:
        marker='.'
    cur_mask=(labels==cur_label)# 获取当前某个数据集的集合的索引
    cur_data=X[cur_mask&mask_core]# 当前数据集中正常的数据
    plt.scatter(cur_data[:,0],cur_data[:,1],marker=marker,edgecolors='black',s=96,facecolors='none')
    cur_data=X[cur_mask&-mask_core]# 当前数据集中异常数据
    plt.scatter(cur_data[:,0],cur_data[:,1],marker=marker,edgecolors='black',s=32)
plt.show()
'''

# 8、近邻传播聚类例子
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
# 生成测试数据
centers = [[1, 1], [-1, -1], [1, -1]]
# 生成实际中心为centers的测试样本300个，X是包含300个(x,y)点的二维数组，labels_true为其对应的真是类别标签
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)

# 计算AP
ap = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = ap.cluster_centers_indices_    # 预测出的中心点的索引，如[123,23,34]
labels = ap.labels_    # 预测出的每个数据的类别标签,labels是一个NumPy数组

n_clusters_ = len(cluster_centers_indices)    # 预测聚类中心的个数

print('预测的聚类中心个数：%d' % n_clusters_)
print('同质性：%0.3f' % metrics.homogeneity_score(labels_true, labels))
print('完整性：%0.3f' % metrics.completeness_score(labels_true, labels))
print('V-值： % 0.3f' % metrics.v_measure_score(labels_true, labels))
print('调整后的兰德指数：%0.3f' % metrics.adjusted_rand_score(labels_true, labels))
print('调整后的互信息： %0.3f' % metrics.adjusted_mutual_info_score(labels_true, labels))
print('轮廓系数：%0.3f' % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# 绘制图表展示
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')    # 关闭所有的图形
plt.figure(1)    # 产生一个新的图形
plt.clf()    # 清空当前的图形

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# 循环为每个类标记不同的颜色
for k, col in zip(range(n_clusters_), colors):
    # labels == k 使用k与labels数组中的每个值进行比较
    # 如labels = [1,0],k=0,则‘labels==k’的结果为[False, True]
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]    # 聚类中心的坐标
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('预测聚类中心个数：%d' % n_clusters_)
plt.show()

# 9、建立客户细分模型
'''
# 1、获取数据
import csv

input_file='wholesale.csv'
file_reader=csv.reader(open(input_file,'r'),delimiter=',')
X=[]
for count,row in enumerate(file_reader):
    if not count:
        names=row[2:]
        continue
    X.append([float(x) for x in row[2:]])
X=np.array(X)
# 2、使用均值漂移训练数据
from sklearn.cluster import MeanShift,estimate_bandwidth

bandwidth=estimate_bandwidth(X,quantile=0.8,n_samples=len(X))
model=MeanShift(bandwidth=bandwidth,bin_seeding=True)
model.fit(X)
labels=model.labels_
clucenters=model.cluster_centers_
num_centers=len(clucenters)
# 3、打印集群中心
print('\t'.join([name[:3] for name in names]))
print('\t'.join([str(clucenter) for clucenter in clucenters]))
# 4、把milk与groceries的聚类结果可视化
centriods_milk_groceries=clucenters[:,1:3]
plt.figure()
plt.scatter(centriods_milk_groceries[:,0],centriods_milk_groceries[:,1],color='k',s=100,facecolor='none')
offset=0.2
plt.xlim(centriods_milk_groceries[:,0].min()-offset*centriods_milk_groceries[:,0].ptp(),centriods_milk_groceries[:,0].max()+offset*centriods_milk_groceries[:,0].ptp())
plt.xlim(centriods_milk_groceries[:,1].min()-offset*centriods_milk_groceries[:,1].ptp(),centriods_milk_groceries[:,1].max()+offset*centriods_milk_groceries[:,1].ptp())
plt.show()
'''


