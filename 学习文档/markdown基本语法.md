# 初级语法
## 换行
1、两个及以上空格+回车     
2、两个回车  
3、使用html中的换行&lt;br/><br/>
## 标题 1-6
* Atx形式  
 # 一级标题  
 ## 二级标题  
 ### 三级标题  

* Setext形式  
 一级标题  
 ===  
 二级标题  
 \___  
## 块区引用  
>\>（大于号）在块中可以其他的语法
## 列表
1. 一朵百合花
2. 两朵百合花
3. 三朵百合花
___
*  无序列表（可用*+-等做标记，标记后有个空格）   
**注意：**无序和有序不能直接连在一块，否则将按照第一行的排序
## 代码区块
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
## 数学公式 

首先打开MarkdownPad2的**工具**->**选项**->**高级**->**html head编辑器**然后添加下列代码保存并关闭。

	<script type="text/javascript"
	   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
	</script>

# 这是行间公式  
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$  
# 这是行内公式  
这里是行内公式 \\(E = mc^2\\) 这里是行内公式  

>`注意：有一些公式不显示，如公式\sum\_{x=0}^{n}显示不出来，但是可以看出\_符号后面都变成斜体字，所以应该在\_前面加\转义即可。`    

* 求和  
\sum\_{x=0}^{n}  对应 \\(\sum\_{x=0}^{n}\\)  
* 极限  
\lim\_{x \to 0} 对应 \\(\lim\_{x \to 0} \\)
* 积分  
\int\_0^\infty{fxdx} 对应 \\(\int\_0^\infty{fxdx} \\)
* 无限  
\infty  对应 \\(\infty\\)  
* 分数形式    
\frac{分子}{分母} 对应 \\(\frac{分子}{分母}\\)  
* 组合数  
C_{2n}^{n}  对应  \\(C\_{2n}^{n}\\)  
* 乘号  
\times 对应  \\(\times\\)  
* 根号  
\sqrt[x]y呈现\\(\sqrt[x]y\\)
* 特殊函数  
\sin x  对应\\(\sin x\\)  
\ln x  对应\\(\ln x\\)  
\max(A,B,C)  对应\\(\max(A,B,C)\\)  
* 希腊字母  
α	\alpha	β	\beta  
γ	\gamma	δ	\delta  
ε	\epsilon	ζ	\zeta  
η	\eta	θ	\theta  
ι	\iota	κ	\kappa  
λ	\lambda	μ	\mu  
ν	\nu	ξ	\xi  
π	\pi	ρ	\rho  
σ	\sigma	τ	\tau  
υ	\upsilon	φ	\phi  
χ	\chi	ψ	\psi  
ω	\omega  
* 特殊符号  
![](https://upload-images.jianshu.io/upload_images/436556-a0a75e713b2b3e9c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/681)
* 矢量  
\vec a 对应 \\(\vec a\\)  
\overrightarrow {xy} 对应 \\(\overrightarrow {xy}\\)
* 字体  
Blackboard Bold：\mathbb {A} 对应 \\(\mathbb {A}\\)
* 分组  
使用{}符号分组，如10^{10}
* 空格  
一个空格：转义符：\ 
四个空格：\quad
* 矩阵  
起始标记\ begin{matrix}，结束标记\ end{matrix}
每一行末尾标记\\\\\\，行间元素之间以&分隔
$$\begin{pmatrix}
1&0&0\\\
0&1&0\\\
0&0&1\\\
\end{pmatrix}$$  
* 矩阵边框  
在起始、结束标记处用下列词替换 matrix
pmatrix ：小括号边框  
bmatrix ：中括号边框  
Bmatrix ：大括号边框  
vmatrix ：单竖线边框  
Vmatrix ：双竖线边框  
* 省略元素    
横省略号：\cdots   
竖省略号：\vdots  
斜省略号：\ddots  
* 列阵  
\ begin{array}{c|ll}  
{↓}&{a}&{b}&{c}\\\\\\  
\hline
{R\_1}&{c}&{b}&{a}\\\\\\  
{R\_2}&{b}&{c}&{c}\\\\\\  
\ end{array}   
对应  

\begin{array}{c|ll}  
{↓}&{a}&{b}&{c}\\\
\hline
{R\_1}&{c}&{b}&{a}\\\
{R\_2}&{b}&{c}&{c}\\\
\end{array}

* 方程式   
\ begin{cases}  
a\_1x+b\_1y+c\_1z=d\_1\\\\\\  
a\_2x+b\_2y+c\_2z=d\_2\\\\\\  
a\_3x+b\_3y+c\_3z=d\_3\\\\\\  
\ end{cases}  
对应

\begin{cases}  
a\_1x+b\_1y+c\_1z=d\_1\\\ 
a\_2x+b\_2y+c\_2z=d\_2\\\
a\_3x+b\_3y+c\_3z=d\_3\\\
\end{cases}    

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

参考：<https://www.jianshu.com/p/a0aa94ef8ab2>