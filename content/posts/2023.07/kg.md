---
title: "Knowledge Graph Reasoning"
date: 2023-07-27T15:04:29+08:00
draft: true
mathjax: true
bibFile: bib/bib.json
---

论文发现于 [Awesome-Knowledge-Graph-Reasoning Github 仓库地址](https://github.com/LIANGKE23/Awesome-Knowledge-Graph-Reasoning)。

## InGram: Inductive Knowledge Graph Embedding via Relation Graphs

{{< cite "9dxgnN5q" >}} 研究了知识图谱的归纳式推理问题。

### 知识图谱

知识图谱是一种多关系图（Multi-relational Graph），包含实体（Entity）作为顶点，关系（Relation）作为边，三元组（Triplet）描述起始顶点通过某类型的连边指向目标顶点。

### 问题设置

现有方法有针对归纳式的实体推理进行研究，即预测训练时未出现的实体；但未针对关系的归纳式推理进行研究。如下图所示。

<img src="iShot_2023-07-27_15.47.54.png" />

本文考虑两种更现实的问题设置：

- **半归纳式关系推理**：推理时存在已知和未知的关系
- **归纳式关系推理**：推理时只有未知的关系

在本文中，给定了两种图谱，即训练图$G_{tr}$和推理图$G_{inf}$。训练图定义为$G_{tr}=(V_{tr},R_{tr},E_{tr})$，其中$V,R,E$分别表示实体、关系和三元组的集合。将$E_{tr}$进一步划分为$E_{tr}=F_{tr}\cup T_{tr}$，其中$F_{tr},T_{tr}$分别表示训练三元组和验证三元组。
类似地，推理图定义为$G_{inf}=(V_{inf},R_{inf},E_{inf})$。将推理图中的三元组集合划分为三个子集：$E_{inf}=F_{inf}\cup T_{val}\cup T_{test}$，其中$F_{inf}, T_{val}, T_{test}$分别表示观测集、验证集和测试集。$F_{inf}$涵盖了$G_{inf}$中所有类型的实体和关系。
在归纳式推理的设定下$V_{tr}\cap V_{inf}=\emptyset$。
$R_{inf}$不一定是$R_{tr}$的子集。

模型首先在$G_{tr}=(V_{tr},E_{tr},F_{tr})$上进行训练，任务是预测$T_{tr}$中的三元组；
在需要验证集调整模型的超参数时，使用$G_{inf}=(V_{inf},R_{inf},F_{inf})$计算表征，以在$T_{val}$上预测的效果作为指标；
推理阶段观察模型在$T_{test}$上的性能。

### 解决方案：InGram

给定图谱$G=(V,R,F)$，对于每个三元组$(v_i,r_k,v_j)\in F$，添加反向关系$r_k^{-1}$到$R$，并添加反向三元组$(v_j,r_k^{-1},r_i)$到$F$。假设实体个数和关系个数分别为$n,m$。
依据图谱构建**以关系为顶点，以边表示关系相似度**的关系图。

#### 构建关系图

设矩阵$E_h\in\mathbb{R}^{n\times m},E_t\in\mathbb{R}^{n\times m}$，其中$E_h[i,j],E_t[i,j]$分别表示实体$v_i$作为关系$r_j$的起始顶点和目标顶点的频率。接着，定义相应的两个邻接矩阵$A_h,A_t$为

$$
\begin{aligned}
A_h &=E_h^TD_h^{-2}E_h\in\mathbb{R}^{m\times m} \\\
A_t &=E_t^TD_t^{-2}E_t\in\mathbb{R}^{m\times m} \\\
\end{aligned}
$$

其中$D_h,D_t$分别表示$E_t,E_h$的度矩阵，即$D_h[i,i]=\sum_jE_h[i,j],D_t$同理。度矩阵起到对矩阵$A_h,A_t$中每个元素归一化的作用。
这样，关系图的邻接矩阵定义为$A=A_h+A_t$。如下图所示。注意到生成的关系图并不是全连接图。

<img src="iShot_2023-07-27_20.35.52.png" />

**InGram框架包括基于图谱对实体的聚合，以及基于关系图对关系的聚合。**

#### 关系聚合

设每个实体的初始特征向量为$x_i\in\mathbb{R}^d$，对$x_i$进行Glorot初始化。那么有如下的聚合公式：

$$
\begin{align}
z\_i^{(0)} &=Hx\_i &&\in\mathbb{R}^{d'} \label{eq:1}\tag{1}\\\
z_i^{(l+1)} &=\sigma(\sum_{r_j\in \mathcal{N}\_i}\alpha_{ij}^{(l)}W^{(l)}z_j^{(l)}) &&\in\mathbb{R}^{d'} \label{eq:2}\tag{2}\\\
\alpha_{ij}^{(l)} &=\frac{\exp\left(y^{(l)}\sigma(P^{(l)}[z_i^{(l)}\parallel z_j^{(l)}])+c_{s(i,j)}^{(l)}\right)}{\sum_{r_{j'}\in \mathcal{N}\_i}\exp\left(y^{(l)}\sigma(P^{(l)}[z_i^{(l)}\parallel z_{j'}^{(l)}])+c_{s_(i,j')}^{(l)}\right)} \label{eq:3}\tag{3}\\\
s(i,j) &=\left\lceil \frac{\text{rank}(a_{i,j})\times B}{\text{nnz}(A)} \right\rceil \label{eq:4}\tag{4}
\end{align}
$$

其中$z_i^{(l)}$表示表征向量，$\mathcal{N}_i$表示关系$r_i$在关系图中的邻居关系集合。$H,W,P$均为可训练的权重矩阵。由($\ref{eq:2}$)得知关系表征向量遵循GAT的聚合方式，$\alpha\_{ij}$为注意力系数。$y^{(l)}\in\mathbb{R}^{1\times d}$是权重系数，用于缓解GAT的固定注意力问题。$1\le s(i,j)\le B$是下标。作者将所有的关系对$\\{(r_i,r_j)\\}$依照相似度划分为$B$个桶。$c\_{\cdot}^{(l)}$是权重系数。

($\ref{eq:3}$)的前半部分是与GAT相似的局部注意力计算方法，后面的$c\_{s(i,j)}^{(l)}$是全局注意力。作者认为从相似度较高的顶点聚合信息是有益的，其实是提出了一种利用关系图原有边权的一种机制。$c_{s(i,j)}^{(l)}$是训练得来的，理想情况下对于原始边权大的关系对$(r_i,r_j)$，学习到的$c_{s(i,j)}^{(l)}$也应该大。作者在实验部分验证了这一点。

经过$L$层GAT最终得到每个关系的表征向量$z_i^{(L)}$。

#### 实体聚合

实体聚合需要基于上面关系聚合的结果$z_i^{(L)}$。设顶点的初始特征为$x_i\in\mathbb{R}^{d_2}$，仍然对$x_i$进行Glorot初始化。那么有如下的聚合公式：

$$
\begin{align}
h_i^{(0)} &= Hx_i \label{eq:5}\tag{5}\\\
z_i^{(L)} &= \sum_{v_j\in\mathcal{N}\_i}\sum_{r_k\in R_{ji}}\frac{z_k^{(L)}}{\sum_{v_j\in \mathcal{N}\_i}|R_{ji}|} \label{eq:6}\tag{6}\\\
h_{i}^{(l+1)} &= \sigma\left(\beta_{ii}^{(l)}W^{(l)}[h_i^{(l)}\parallel z_i^{(L)}]+\sum_{v_j\in\mathcal{N}\_i}\sum_{r_k\in R_{ji}}\beta_{ijk}^{(l)}W^{(l)}[h_j^{(l)}\parallel z_k^{(L)}]\right) \label{eq:7}\tag{7}\\\
\beta_{ii}^{(l)} &= \frac{\exp(y^{(l)}\sigma(P^{(l)}b_{ii}^{(l)}))}{\exp(y^{(l)}\sigma(P^{(l)}b_{ii}^{(l)}))+\sum_{v_{j'}\in\mathcal{N}\_i}\sum_{r_{k'}\in R_{j'i}}\exp(y^{(l)}\sigma(P^{(l)}b_{ij'k'}^{(l)}))} \label{eq:8}\tag{8}\\\
\beta_{ijk}^{(l)} &= \frac{\exp(y^{(l)}\sigma(P^{(l)}b_{ijk}^{(l)}))}{\exp(y^{(l)}\sigma(P^{(l)}b_{ii}^{(l)}))+\sum_{v_{j'}\in\mathcal{N}\_i}\sum_{r_{k'}\in R_{j'i}}\exp(y^{(l)}\sigma(P^{(l)}b_{ij'k'}^{(l)}))} \label{eq:9}\tag{9}\\\
\end{align}
$$

其中$h_i$表示实体$v_i$的表征向量。$\mathcal{N}\_i=\\{v_j|(v_j,r_k,v_i)\in F,v_j\in V,r_k\in R\\}$表示实体$v_i$的邻居顶点集合，$R_{ji}$表示实体$v_j\to v_i$的关系集合。$W,P$仍然表示可训练的权重系数。
由($\ref{eq:7}$)式可知实体从上一轮自己的表征和邻居实体的表征聚合信息。$\beta_{ii},\beta{ijk}$表示注意力系数。$b_{ii}^{(l)}=[h_i^{(l)}\parallel h_i^{(l)} \parallel z_i^{(L)}],b_{ijk}^{(l)}=[h_i^{(l)}\parallel h_j^{(l)} \parallel z_k^{(L)}]$。

经过$L$层聚合后得到实体表征$h_i^{(L)}$。