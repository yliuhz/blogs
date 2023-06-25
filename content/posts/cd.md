---
title: "Structural Community Detection"
date: 2023-06-19T10:55:48+08:00
draft: false
mathjax: true
bibFile: bib/bib.json
---

本文专注于解释社区发现的两个经典算法：Louvain方法和Infomap方法。

## 问题定义

给定一个图$G=(V,E)$，找到一个映射$g:V\to \{1,2,\cdots,K\}$，$g$将图中的顶点映射到社区标签。

## Louvain方法

Louvain方法 {{< cite "2hXkUYg3" >}} 是一种贪心算法，其优化目标是模块度，如下式所示：

$$Q=\frac{1}{2m}\sum_{i,j}\left[A_{ij}-\frac{k_ik_j}{2m}\right]\delta(c_i,c_j)$$

其中$A_{ij}$表示顶点$i$和顶点$j$之间连边的权重；$k_i=\sum_jA_{ij}$表示与顶点$i$相连的边的权重之和；$c_i$表示顶点$i$的社区标签；$\delta(u,v)$在$u=v$时等于1，否则等于0；$m=\frac{1}{2}\sum_{ij}A_{ij}$。

Louvain算法分为两阶段。
```
初始时设定每个顶点独立属于一个社区
# 第一阶段
生成一个随机的顶点序列Queue
For each node i in Queue:
    For each neighbor j of i:
        尝试将i的社区标签c_i修改为j的社区标签c_j
        计算模块度的增长DQ
    If max(DQ)>0:
        修改i的社区标签
    Else:
        保持i的社区标签不变
# 第二阶段
For each community c_i in G:
    将社区标签为c_i的所有顶点聚合为一个新的顶点
    原c_i内的连边转化为新顶点的自环，边权为原边权之和
    原c_i内顶点与另一社区c_j内顶点的所有连边聚合为一条连边，边权为原边权之和
# 重新执行第一阶段，直到模块度Q不再增加
```

第一阶段中顶点序列的顺序会影响算法的输出。作者发现不同的顶点处理顺序会影响算法的时间效率，但不会对最终的模块度造成过大（significant）的影响。

## Infomap方法

Infomap方法 {{< cite "umB3JCfk" >}} 与Louvain方法相似，但使用了不同的优化目标。作者发现了社区发现与编码的联系，即最优的社区发现结果能够使得描述图上随机游走路径的编码长度最短。如下图所示。当未做社区划分时，对于图(a)中的随机游走，需要在图(b)中使用所有路径上顶点的编码进行描述，一共$314$bits；当把图划分成四个社区后，在图(c)中，为每个社区设定起始编码和中止编码，路径节点的编码长度可以缩小，这样整条游走路径的编码长度缩短为$243$bits；在图(d)中，若忽略社区内部的游走路径，则可以产生更短的的粗糙编码。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/iShot_2023-06-20_10.36.17.png" />

对于一个含有$n$个节点的图，可以将其划分为$1,2,\cdots,n$个社区，即一共有$n$种社区个数的选择（下图中为$1,2,\cdots,25$共25种选择）。在比较不同社区划分对编码长度的影响时，作者发现，描述社区的平均编码长度随着社区个数的增加而单调递增，描述节点的平均编码长度随着社区个数的增加而单调递减。将二者相加即为描述游走路径的平均编码长度。当产生最优的社区个数($4$)时，平均编码长度最短($3.09$bits)

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/iShot_2023-06-20_10.36.29.png" />

具体来说，Infomap的优化目标是最小化如下的Map Equation。

$$L(M)=q_{\curvearrowright}H(Q)+\sum_{i=1}^mp_{\circlearrowright}^iH(P^i)$$

其中$H(Q),H(P^i)$分别描述节点编码的平均长度和描述社区$i$的编码平均长度。$q_{i\curvearrowright}$表示离开社区$i$的概率，$q_{\curvearrowright}=\sum_{i=1}^mq_{i\curvearrowright}$表示随机游走切换社区的概率。$p_{\alpha}$表示访问节点$\alpha$的概率，$p_{\circlearrowright}=\sum_{\alpha\in i}p_{\alpha}+q_{i\curvearrowright}$表示访问及离开社区$i$的概率之和。


它基于香农源编码定理：当使用$n$个编码描述一个随机变量$X$的$n$种状态时，若每个状态$i$出现的概率为$p_i$，则该编码的长度不能低于随机变量$X$自身的熵：$H(X)=-\sum_{i=1}^np_i\log(p_i)$。
