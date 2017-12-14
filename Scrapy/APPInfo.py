# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:39:12 2017

@author: Fangzhifeng
"""
import urllib.request
import re

#豌豆荚App信息
url='http://www.wandoujia.com/apps'#豌豆荚首页链接

def getAllLinks(url):#获取首页链接的所有子链接
    html1=str(urllib.request.urlopen(url).read())
    pat='<a class="cate-link" href="(http://.+?")>'
    allLink=re.compile(pat).findall(html1)
    allLinks=[]
    for link in allLink:
        allLinks.append(link.split('"')[0])
    return allLinks
#print(getAllLinks(url))
def getAllDescLinks(url,page):#获取子链接中所有app指向的链接
    url=url+'/'+str(page)
    print(url)
    html1=str(urllib.request.urlopen(url).read().decode('utf-8'))
    pat2='<ul id="j-tag-list" class="app-box clearfix">[\s\S]*<div class="pagination">'
    allLink=str(re.compile(pat2).findall(html1)).strip('\n').replace(' ','').replace('\\n','').replace('\\t','')
    allLink=allLink.split('<divclass="icon-wrap"><ahref="')
    allLinks=[]
    for i in range(1,len(allLink)):
        allLinks.append(allLink[i].split('"><imgsrc')[0])
    allLinks=list(set(allLinks))
    return allLinks
#string = getAllDescLinks('http://www.wandoujia.com/category/5029',3)
#getAllInfo('http://www.wandoujia.com/apps/com.kugou.android')
def getAppName(html):#获取app名字
    pat='<span class="title" itemprop="name">[\s\S]*</span>'
    string=str(re.compile(pat).findall(html))
    name=''
    if string!='[]':
        name=string.split('>')[1].split('<')[0]
    return name
def getDownNumber(html):#下载次数
    pat='<i itemprop="interactionCount"[\s\S]*</i>'
    string=str(re.compile(pat).findall(html))
    num=''
    if string!='[]':
        num=string.split('>')[1].split('<')[0]
    return num
def getScore(html):#评分
    pat='<span class="item love">[\s\S]*<i>[\s\S]*好评率</b>'
    string=str(re.compile(pat).findall(html))
    score=''
    if string!='[]':
        score=string.split('i')[2].split('>')[1].split('<')[0]
    return score
def getIconLink(html):#app中icom的图片链接
    pat='<div class="app-icon"[\s\S]*</div>'
    image=str(re.compile(pat).findall(html))
    img=''
    if image!='[]':
        img='http://'+str(image).split('http://')[1].split('.png')[0]+'.png'
    return img
def getVersion(html):#版本
    pat='版本</dt>[\s\S]*<dt>要求'
    version=str(re.compile(pat).findall(html))
    if version!='[]':
        version=version.split('&nbsp;')[1].split('</dd>')[0]
    return version
def getSize(html):#大小
    pat='大小</dt>[\s\S]*<dt>分类'
    size=str(re.compile(pat).findall(html))
    if size!='[]':
        size=size.split('<dd>')[1].split('<meta')[0].strip('\n').replace(' ','').replace('\\n','')#strip删除本身的换行，删除中文的空格，删除\n字符
    return size

def getImages(html):#所有截屏的链接
    pat='<div data-length="5" class="overview">[\s\S]*</div>'
    images1=str(re.compile(pat).findall(html))
    pat1='http://[\s\S]*.jpg'
    images=[]
    images1=str(re.compile(pat1).findall(images1))
    if images1!='[]':
        images1=images1.split('http://')
        for i in range(1,len(images1)):
            images.append(images1[i].split('.jpg')[0]+'.jpg')
    return images
def getAbstract(html):#简介
    pat='<div data-originheight="100" class="con" itemprop="description">[\s\S]*<div class="change-info">'
    abstract=str(re.compile(pat).findall(html))
    if abstract=='[]':
        pat='<div data-originheight="100" class="con" itemprop="description">[\s\S]*<div class="all-version">'
        abstract=str(re.compile(pat).findall(html))
    if abstract!='[]':
        abstract=abstract.split('description">')[1].split('</div>')[0].replace('<br>','').replace('<br />','')#strip删除本身的换行，删除中文的空格，删除\n字符
    return abstract
def getUpdateTime(html):#更新时间
    pat='<time id="baidu_time" itemprop="datePublished"[\s\S]*</time>'
    updateTime=str(re.compile(pat).findall(html))
    if updateTime!='[]':
        updateTime=updateTime.split('>')[1].split('<')[0]
    return updateTime
def getUpdateCon(html):#更新内容
    pat='<div class="change-info">[\s\S]*<div class="all-version">'
    update=str(re.compile(pat).findall(html))
    if update!='[]':
        update=update.split('"con">')[1].split('</div>')[0].replace('<br>','').replace('<br />','')#strip删除本身的换行，删除中文的空格，删除\n字符
    return update
def getCompany(html):#开发公司
    pat='<span class="dev-sites" itemprop="name">[\s\S]*</span>'
    com=str(re.compile(pat).findall(html))
    if com!='[]':
        com=com.split('"name">')[1].split('<')[0]#strip删除本身的换行，删除中文的空格，删除\n字符
    return com 
def getClass(html):#所属分类
    pat='<dd class="tag-box">[\s\S]*<dt>TAG</dt>'
    classfy1=str(re.compile(pat).findall(html))
    classfy=[]
    if classfy1!='[]':
        classfy1=classfy1.split('appTag">')
        for i in range(1,len(classfy1)):
            classfy.append(classfy1[i].split('<')[0])
    return classfy 
def getTag(html):#标有的Tag
    pat='<div class="side-tags clearfix">[\s\S]*<dt>更新</dt>'
    tag1=str(re.compile(pat).findall(html))
    tag=[]
    if tag1!='[]':
        tag1=tag1.strip('\n').replace(' ','').replace('\\n','').split('</a>')
        for i in range(0,len(tag1)-1):
            tag.append(tag1[i].replace('<divclass="side-tagsclearfix">','').replace('<divclass="tag-box">','').replace('</div>','').split('>')[1])
    return tag 
def getDownLink(html):#下载链接
    pat='<div class="qr-info">[\s\S]*<div class="num-list">'
    link=str(re.compile(pat).findall(html))
    if link!='[]':
        link=link.split('href="http://')[1].split('" rel="nofollow"')[0]
    return link 
def getComment(html):#评论内容（只包含10条，因为网页只显示有限）
    pat='<ul class="comments-list">[\s\S]*<div class="hot-tags">'
    comm=str(re.compile(pat).findall(html))
    comms=''
    eval_descs=[]
    if comm!='[]':
        comms=comm.strip('\n').replace(' ','').replace('\\n','').split('<liclass="normal-li">')
        for i in range(1,len(comms)-1):
            userName=comms[i].split('name">')[1].split('<')[0]
            time=comms[i].split('</span><span>')[1].split('<')[0]
            evalDesc=comms[i].split('content"><span>')[1].split('<')[0]
            eval_desc={'userName':userName,'time':time,'evalDesc':evalDesc}
            eval_descs.append(eval_desc)
    # comm=comm.split('href="http://')[1].split('" rel="nofollow"')[0]
    return eval_descs
def getAllInfo(url):#获取所有信息
    html1=str(urllib.request.urlopen(url).read().decode('utf-8'))
    name=getAppName(html1)
    print('名称:',name)
    if name=='':
        return 
    num=str(getDownNumber(html1))
    print('下载次数:',num)
    icon=str(getIconLink(html1))
    print('log链接:',icon)
    score=str(getScore(html1))
    print('评分:',score)
    version=str(getVersion(html1))
    print('版本:',version)
    size=str(getSize(html1))
    print('大小:',size)
    images=str(getImages(html1))
    print('截图:',images)
    abstract=str(getAbstract(html1))
    print("简介:",abstract)
    updateTime=str(getUpdateTime(html1))
    print('更新时间:',updateTime)
    updateCon=str(getUpdateCon(html1))
    print('更新内容:',updateCon)
    com=str(getCompany(html1))
    print('公司:',com)
    classfy=str(getClass(html1))
    print('分类:',classfy)
    tag=str(getTag(html1))
    print('Tag:',tag)
    downLink=str(getDownLink(html1))
    print('下载链接:',downLink)
    comm=str(getComment(html1))
    print('评价:',comm)
    if name!='':
        insertAllInfo(name,num,icon,score,version,size,images,abstract,updateTime,updateCon,com,classfy,tag,downLink,comm)

def insertAllInfo(name,num,icon,score,appversion,size,images,abstract,updateTime,updateCon,com,classfy,tag,downLink,comm):#插入SQL数据库
    import pyodbc
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=127.0.0.1,1433;DATABASE=Test;UID=sa;PWD=123')  
    #连接之后需要先建立cursor：
    cursor = conn.cursor()
    try:
        cursor = conn.cursor()
        cursor.execute('insert into tb_wandoujia(name,num,icon,score,appversion,size,images,abstract,updateTime,updateCon,com,classfy,tag,downLink,comm) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(name,num,icon,score,appversion,size,images,abstract,updateTime,updateCon,com,classfy,tag,downLink,comm))
        conn.commit()# 不执行不能插入数据
        print('成功')
    except Exception as e:
        print(str(e))
    finally:
        conn.close()


for link in getAllLinks(url):
    print(link)
    for i in range(1,42):#由于豌豆荚给的最大是42页，所以这里用42，反正如果没有42，也会很快
        print(i)
        for descLink in getAllDescLinks(link,i):
            print(descLink)
            getAllInfo(descLink)
            

#getAllInfo('http://www.wandoujia.com/apps/com.huawei.hwvplayer')










