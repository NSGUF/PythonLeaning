# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:39:12 2017

@author: Fangzhifeng
"""
import urllib.request
import re
#豌豆荚App信息
url='http://www.wandoujia.com/apps'

def getAllLinks(url):
    html1=str(urllib.request.urlopen(url).read())
    pat='<a class="cate-link" href="(http://.+?")>'
    allLink=re.compile(pat).findall(html1)
    allLinks=[]
    for link in allLink:
        allLinks.append(link.split('"')[0])
    return allLinks
#print(getAllLinks(url))
def getAllDescLinks(url,page):
    html1=str(urllib.request.urlopen(url).read())
    pat2='<a href="(http://.+?)"'
    allLink=re.compile(pat2).findall(html1)
    allLinks=[]
    for link in allLink:
        if re.search('/apps/',link)!=None:
            allLinks.append(link)
    allLinks=list(set(allLinks))
    return allLinks
def getAppName(html):
    pat='<span class="title" itemprop="name">[\s\S]*</span>'
    string=str(re.compile(pat).findall(html))
    name=string.split('>')[1].split('<')[0]
    return name
def getDownNumber(html):
    pat='<i itemprop="interactionCount"[\s\S]*</i>'
    string=str(re.compile(pat).findall(html))
    num=string.split('>')[1].split('<')[0]
    return num
def getScore(html):
    pat='<span class="item love">[\s\S]*<i>[\s\S]*好评率</b>'
    string=str(re.compile(pat).findall(html))
    num=string.split('i')[2].split('>')[1].split('<')[0]
    return num
def getIconLink(html):
    pat='<div class="app-icon"[\s\S]*</div>'
    image=str(re.compile(pat).findall(html))
    img='http://'+str(image).split('http://')[1].split('.png')[0]+'.png'
    return img
def getVersion(html):
    pat='版本</dt>[\s\S]*<dt>要求'
    version=str(re.compile(pat).findall(html))
    version=version.split('&nbsp;')[1].split('</dd>')[0]
    return version
def getSize(html):
    pat='大小</dt>[\s\S]*<dt>分类'
    version=str(re.compile(pat).findall(html))
    version=version.split('<dd>')[1].split('<meta')[0].strip('\n').replace(' ','').replace('\\n','')#strip删除本身的换行，删除中文的空格，删除\n字符
    return version

def getImages(html):
    pat='<div data-length="5" class="overview">[\s\S]*</div>'
    images1=str(re.compile(pat).findall(html))
    pat1='http://[\s\S]*.jpg'
    images=[]
    images1=str(re.compile(pat1).findall(images1)).split('http://')
    for i in range(1,len(images1)):
        images.append(images1[i].split('.jpg')[0]+'.jpg')
    return images
def getAbstract(html):
    pat='<div data-originheight="100" class="con" itemprop="description">[\s\S]*<div class="change-info">'
    abstract=str(re.compile(pat).findall(html))
    abstract=abstract.split('description">')[1].split('</div>')[0].replace('<br>','').replace('<br />','')#strip删除本身的换行，删除中文的空格，删除\n字符
    return abstract
def getUpdateTime(html):
    pat='<time id="baidu_time" itemprop="datePublished"[\s\S]*</time>'
    updateTime=str(re.compile(pat).findall(html))
    updateTime=updateTime.split('>')[1].split('<')[0]
    return updateTime
def getUpdateCon(html):
    pat='<div class="change-info">[\s\S]*<div class="all-version">'
    update=str(re.compile(pat).findall(html))
    update=update.split('"con">')[1].split('</div>')[0].replace('<br>','').replace('<br />','')#strip删除本身的换行，删除中文的空格，删除\n字符
    return update
def getCompany(html):
    pat='<span class="dev-sites" itemprop="name">[\s\S]*</span>'
    com=str(re.compile(pat).findall(html))
    com=com.split('"name">')[1].split('<')[0]#strip删除本身的换行，删除中文的空格，删除\n字符
    return com 
def getClass(html):
    pat='<dd class="tag-box">[\s\S]*<dt>TAG</dt>'
    classfy1=str(re.compile(pat).findall(html))
    classfy1=classfy1.split('appTag">')
    classfy=[]
    for i in range(1,len(classfy1)):
        classfy.append(classfy1[i].split('<')[0])
    return classfy 
def getTag(html):
    pat='<div class="side-tags clearfix">[\s\S]*<dt>更新</dt>'
    tag1=str(re.compile(pat).findall(html)).strip('\n').replace(' ','').replace('\\n','')
    tag1=tag1.split('</a>')
    tag=[]
    for i in range(0,len(tag1)-1):
        tag.append(tag1[i].replace('<divclass="side-tagsclearfix">','').replace('<divclass="tag-box">','').replace('</div>','').split('>')[1])
    return tag 
def getDownLink(html):
    pat='<div class="qr-info">[\s\S]*<div class="num-list">'
    link=str(re.compile(pat).findall(html))
    link=link.split('href="http://')[1].split('" rel="nofollow"')[0]
    return link 
def getComment(html):
    pat='<ul class="comments-list">[\s\S]*<div class="hot-tags">'
    comm=str(re.compile(pat).findall(html)).strip('\n').replace(' ','').replace('\\n','')
    comms=comm.split('<liclass="normal-li">')
    eval_descs=[]
    for i in range(1,len(comms)-1):
        userName=comms[i].split('name">')[1].split('<')[0]
        time=comms[i].split('</span><span>')[1].split('<')[0]
        evalDesc=comms[i].split('content"><span>')[1].split('<')[0]
        eval_desc={'userName':userName,'time':time,'evalDesc':evalDesc}
        eval_descs.append(eval_desc)
    # comm=comm.split('href="http://')[1].split('" rel="nofollow"')[0]
    return eval_descs
def getAllInfo(url):
    html1=str(urllib.request.urlopen(url).read().decode('utf-8'))
    name=getAppName(html1)
    print('名称:',name)
    num=getDownNumber(html1)
    print('下载次数:',num)
    icon=getIconLink(html1)
    print('log链接:',icon)
    score=getScore(html1)
    print('评分:',score)
    version=getVersion(html1)
    print('版本:',version)
    size=getSize(html1)
    print('大小:',size)
    images=getImages(html1)
    print('截图:',images)
    abstract=getAbstract(html1)
    print("简介:",abstract)
    updateTime=getUpdateTime(html1)
    print('更新时间:',updateTime)
    updateCon=getUpdateCon(html1)
    print('更新内容:',updateCon)
    com=getCompany(html1)
    print('公司:',com)
    classfy=getClass(html1)
    print('分类:',classfy)
    tag=getTag(html1)
    print('Tag:',tag)
    downLink=getDownLink(html1)
    print('下载链接:',downLink)
    comm=getComment(html1)
    print('评价:',comm)


getAllInfo('http://www.wandoujia.com/apps/com.kugou.android')

#==============================================================================
# for link in getAllLinks(url):
#     print(link)
#     for i in range(1,42):
#         print(i)
#         for descLink in getAllDescLinks(link,i):
#             print(descLink)
#             getAllInfo(descLink)
#==============================================================================













