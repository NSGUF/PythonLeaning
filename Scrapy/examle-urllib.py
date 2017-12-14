# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:44:19 2017

@author: ZhifengFang
"""

# Urllib的使用
import urllib.request

# ==============================================================================
# 
# file=urllib.request.urlopen('http://www.baidu.com',timeout=1)#获取整个页面的html,设置超时时间
# data=file.read()
# #print(data)
# dataline=file.readline()
# 
# 
# fhandle=open('E:/1.html','wb')
# fhandle.write(data)
# fhandle.close()
# filename=urllib.request.urlretrieve('http://edu.51cto.com',filename='E:/2.html')#将图片都换成网站中的图片链接，带http的
# urllib.request.urlcleanup()
# print(file.info())#相关信息
# print(file.getcode())#状态码
# print(file.geturl())#url
# print(urllib.request.quote('http://ww.sina.com.cn'))#编码
# print(urllib.request.unquote('http%3A//ww.sina.com.cn'))#解码
# 
# ==============================================================================
# ==============================================================================
# #浏览器的模拟 Header属性
# url='http://download.csdn.net/download/nsguf/9851189'
# file=urllib.request.urlopen('http://blog.csdn.net/weiwei_pig/article/details/51178226')
# print(file.read())
# ==============================================================================

# ==============================================================================
# for i in range(100):
#     try:
#         file=urllib.request.urlopen('https://www.baidu.com/s?wd=csdn&rsv_spt=1&rsv_iqid=0x80fe04d80001f1fe&issp=1&f=8&rsv_bp=0&rsv_idx=2&ie=utf-8&tn=baiduhome_pg&rsv_enter=1&rsv_sug3=6&rsv_sug1=6&rsv_sug7=100&rsv_sug2=0&inputT=3238&rsv_sug4=5708',timeout=0.2)
#         data=file.read()
#         print(len(data))
#     except Exception as e:
#         print("异常:",e)
# ============================================================= =================

#==============================================================================
#  # Get请求实例分析
#  keywd = '方智峰'
#  url = 'http://www.baidu.com/s?wd=' + urllib.request.quote(keywd)
#  req = urllib.request.Request(url)
#  data = urllib.request.urlopen(req).read()
#  fhandle=open('E://4.html','wb')
#  fhandle.write(data)
#  fhandle.close()
#==============================================================================

#==============================================================================
#  url='http://www.iqianyue.com/mypost'#模拟post请求
#  postData=urllib.parse.urlencode({
#      'name':'呵呵',
#      'pass':'123'
#  }).encode('gbk')
#  req=urllib.request.Request(url,postData)
#  data=urllib.request.urlopen(req).read()
#  fhandle=open('E:/6.html','wb')
#  fhandle.write(data)
#  fhandle.close()
#==============================================================================

#==============================================================================
#  #设置代理服务器
#  def use_proxy(proxy_addr,url):
#      proxy=urllib.request.ProxyHandler({'http':proxy_addr})
#      opener=urllib.request.build_opener(proxy,urllib.request.HTTPHandler)
#      urllib.request.install_opener(opener)
#      data=urllib.request.urlopen(url).read().decode('utf-8')
#      return data
#  proxy_addr='121.225.85.109:8118'
#  data=use_proxy(proxy_addr,'http://www.baidu.com')
#  print(len(data))
#==============================================================================

#==============================================================================
# #正则表达式
# import re
# pattern='fang'
# string='fangzhifeng'
# result=re.search(pattern,string)
# print(result)
#==============================================================================

#京东商城图片小例子
import re

def craw(url,page):
    html1=urllib.request.urlopen(url).read()
    html1=str(html1)
    print(html1)
    pat1='<img width="220" height="220" data-img="1" src="//(.+?\.jpg)">'
#==============================================================================
#     imagelist=re.compile(pat1).findall(html1)
#     x=1
#     for imageurl in imagelist:
#         imagename='E:/jd/'+str(page)+str(x)+'.jpg'
#         imageurl='http://'+imageurl
#         try:
#             urllib.request.urlretrieve(imageurl,filename=imagename)
#         except urllib.error.URLError as e:
#             if hasattr(e,'code'):
#                 x+=1
#             if hasattr(e,'reason'):
#                 x+=1
#         x+=1
#==============================================================================
for i in range(1,79):
    url='https://list.jd.com/list.html?cat=9987,653,655&page='+str(i)
    craw(url,i)

#==============================================================================
# #多线程爬取数据
# import threading
# class A(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#     def run(self):
#         for i in range(10):
#             print('A')
#             
# class B(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#     def run(self):
#         for i in range(10):
#             print('B')
# 
# t1=A()
# t2=B()
# t1.start()
# t2.start()
#==============================================================================




























