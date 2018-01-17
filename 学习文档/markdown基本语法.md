  

#初级语法
##换行
1、两个及以上空格+回车     
2、两个回车  
3、使用html中的换行&lt;br/><br/>
##标题 1-6
* Atx形式  
 # 一级标题  
 ## 二级标题  
 ### 三级标题  

* Setext形式  
 一级标题  
 ===  
 二级标题  
 \___  
##块区引用  
>\>（大于号）在块中可以其他的语法
##列表
1. 一朵百合花
2. 两朵百合花
3. 三朵百合花
___
*  无序列表（可用*+-等做标记，标记后有个空格）   
**注意：**无序和有序不能直接连在一块，否则将按照第一行的排序
##代码区块
**内行代码块：**一句话中的`代码块`，使用`符号  

```c
这里也是，使用了三个`符号即可
```  

    代码区块（一个tab或者四个空格）
## 分隔线
  * 三个以上的*或-或_
## 区段元素
内行式：在链接[文字](http://example.com/ "显示文字")后添加（链接）  
`在链接[文字](http://example.com/ "显示文字")后添加（链接）  `  
参考式：在链接[文字][id]后添加（链接） 
[id]: http://example.com/ "显示文字"
[id]: http://example.com/ '显示文字'
[id]: http://example.com/ (显示文字)
[id]: <http://example.com/>  "Optional Title Here"
[id]: http://example.com/longish/path/to/resource/here
    "Optional Title Here"  

	在链接[文字][id]后添加（链接） 
	[id]: http://example.com/ "显示文字"
	[id]: http://example.com/ '显示文字'
	[id]: http://example.com/ (显示文字)
	[id]: <http://example.com/>  "Optional Title Here"
	[id]: http://example.com/longish/path/to/resource/here
	    "Optional Title Here"  

**注意：**链接辨别标签不分大小写
##强调
* 粗体：两个*或_包围
* 斜体：一个*或_包围
##图片
    ![图片文字](路径)
    ![图片文字][id]
    [id]: 路径  alt显示
##其他
###自动链接
* <http://www.baidu.com/>  
```<http://www.baidu.com/>```
* [百度](http://www.baidu.com/)  
```[百度](http://www.baidu.com/)``` 
###反斜杠
可帮助插入普通符号  

	\   反斜线
	`   反引号
	*   星号
	_   底线
	{}  花括号
	[]  方括号
	()  括弧
	#   井字号
	+   加号
	-   减号
	.   英文句点
	!   惊叹号
