---
title: "Unsupervised Deep Graph Structure Learning"
date: 2023-06-27T09:59:56+08:00
draft: false
mathjax: true
bibFile: bib/bib.json
---

现实中图的结构可能是不完整或有噪声的。为了在图结构不可靠的情况下较好地完成下游任务，研究者提出了如下的图结构学习算法。

## Towards Unsupervised Deep Graph Structure Learning 

论文链接：{{< cite "14nyanSAU" >}}

### 相关工作 - 深度图结构学习

一些传统机器学习算法，如图信号处理，谱聚类，图论等可以解决图结构学习问题。但这类方法往往不能处理图上的高维属性。

最近的深度图结构学习方法用于提升GNN在下游任务上的性能。它们遵循相似的管线：先使用一组可学习的参数建模图的邻接矩阵，再和GNN的参数一起针对下游任务进行优化。基于图结构离散的特性，有多种建模图结构的方法。

- 概率模型：伯努利概率模型、随机块模型
- 度量学习：余弦相似度、点积
- 直接使用$n\times n$的参数矩阵建模邻接矩阵

### 问题定义

给定输入图$G=(V,E,X)=(A,X)$，$|V|=n,|E|=m,X\in\mathbb{R}^{n\times d}$

- **结构推理问题**：输入信息只有顶点特征矩阵$X$
- **结构修改问题**：输入信息包含了$A,X$，但$A$可能带有噪声

### 解决方案 - SUBLIME

SUBLIME {{< cite "14nyanSAU" >}} 将学习到的图结构视作一种数据增强，与原图进行多视角的对比学习。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/iShot_2023-06-27_10.22.17.png" />

- **锚点视角（教师）**：对于输入带有邻接矩阵$A$的结构修改任务，直接使用输入的$A$作为锚点；对于输入不带邻接矩阵的结构推理任务，使用单位矩阵$I$作为锚点。节点特征使用输入特征$X$。
- **结构学习器视角（学生）**：使用结构学习器的输出作为该视角。节点特征使用输入特征$X$。

SUBLIME定义了四种图结构学习器，使用时需要当作超参数调节：

- **全图参数化学习器FGP**：顾名思义，直接使用一个$n\times n$的参数矩阵作为学习的邻接矩阵，即$S=\sigma(\Omega)$，$\Omega\in\mathbb{R}^{n\times n}$；

- **度量学习学习器**：先通过输入得到节点的表征$E\in\mathbb{R}^{n\times d}$，再由表征构建学习的邻接矩阵：

$$S=\phi(h_w(X,A))=\phi(E)$$

其中$\phi$是非参数函数（即不用训练的函数），如余弦相似度、闵可夫斯基距离（Minkowski distance）等；$h_w$是表征网络。SUBLIME提供了3种得到$E$的表征方法：注意力学习器、MLP学习器和GNN学习器：

- **注意力学习器**：$E^{(l)}=h_w^{(l)}(E^{(l-1)})=\sigma([e_1^{(l-1)}\odot w^{(l)},\cdots,e_n^{(l-1)}\odot w^{(l)}])^T$，其中$E^{(l)}\in\mathbb{R}^{n\times d}$表示第$l$层表征网络的输出；
$e_i^{(l-1)}\in\mathbb{R}^d$表示$E^{(l-1)}$的第$i$行。初始时$E^{(0)}=X$。可以看到该表征学习器没有对特征进行降维，每一层的输出$E$的维度都是$\mathbb{R}^{n\times d}$。

- **MLP学习器**：$E^{(l)}=h_w^{(l)}(E^{(l-1)})=\sigma(E^{(l-1)}\Omega^{(l)})$，其中$\Omega^{(l)}\in\mathbb{R}^{d\times d}$是参数矩阵。
- **GNN学习器**：$E^{(l)}=h_w^{(l)}(E^{(l-1)})=\sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}E^{(l-1)}\Omega^{(l)})$，其中$\Omega^{(l)}\in\mathbb{R}^{d\times d}$是参数矩阵。

图学习器得到的$S$是全连接的，且不一定对称，所以需要进行后处理：即**稀疏化**、**对称化**和**正则化**。

- **稀疏化**采用$k$最近邻方法，对每个顶点只保留最近的$k$的邻居。
- **对称化**：$S^{sym}=(\sigma(S)+\sigma(S)^T)/2$，其中$\sigma$是ReLU或ELU函数，保证$S'$内的元素都是非负数。
- **正则化**：为保证边权都位于$[0,1]$范围，使用常用的图正则方法：$S'=\tilde{D}^{-1/2}\tilde{S}^{sym}\tilde{D}^{-1/2}$。

得到两个视角的图（锚点视角和学习器视角）后，SUBLIME对每个视角做进一步的数据增强，包括特征扰动和边的扰动。接着，使用GNN编码器（这里使用GCN）将两个视角的图映射到欧氏空间，再使用MLP进一步映射，在MLP的输出上使用节点级别的对比学习损失函数，即最大化同一节点在两个视角图之间的互信息：

$$\begin{aligned}\mathcal{L} &=\frac{1}{2n}\left[l(z_{l,i},z_{a,i})+l(z_{a,i},z_{l,i})\right]\\\ 
l(z_{l,i},z_{a,i}) &= \log\frac{\exp(\cos(z_{l,i},z_{a,i})/t)}{\sum_{k=1}^n\exp(\cos(z_{l,i},z_{a,k})/t)}\end{aligned}$$

其中$t$是温度超参数。

## SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks

论文链接：{{< cite "19EFWTVYY" >}}

### 相关工作

作者罗列了图结构学习的可能方法：

- **相似度矩阵**：根据节点之间的相似度，使用$k$最近邻等方法将节点与最相近的$k$个邻居节点相连。
- **全连接图**：
- **图学习**：
- **领域知识**：

### 问题定义

