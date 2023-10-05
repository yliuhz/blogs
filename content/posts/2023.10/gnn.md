---
title: "A Review of Graph Encoders"
date: 2023-10-04T16:40:22+08:00
draft: true
mathjax: true
bibFile: bib/bib.json
---

本文致力于梳理常见的图节点编码器，包括较早的DeepWalk, Node2vec以及各种经典的图神经网络。

## DeepWalk: Online Learning of Social Representations

{{< cite "PGy3jyD8" >}}提出了DeepWalk。

### 相关定义和记号

给定图$G=(V,E)$，以及其对应的标记图$G_L=(V,E,X,Y)$，其中$X\in \mathbb{R}^{|V|\times S}$表示初始属性矩阵，$Y\in\mathbb{R}^{|V|\times |Y|}$表示节点的标签矩阵。表征学习的目标是得到低维表征矩阵$X_E\in\mathbb{R}^{|V|\times d}$，其中$d<S$，使得表征向量具有如下性质：

- 适应性（Adaptability）：真实的社交网络可能是不断变化的，在图中插入新的边时不应当重新执行表征算法；
- 反映社区结构（Community Aware）
- 低维（Low Dimentional）
- 连续（Continuous）：表征空间是连续空间，即每个元素是实数。

#### 随机游走（Random Walk）

将从节点$v_i$开始的游走路径记为$W_{v_i}$，整条路径可以表示为

$$W_{v_i}=\{W_{v_i}^1, W_{v_i}^2, \cdots, W_{v_i}^k, W_{v_i}^{k+1}, \cdots\}$$
其中$W_{v_i}^{k+1}$为随机挑选的节点$W_{v_i}^k$的邻居。
随机游走方法已被应用于内容推荐问题和社区发现问题中（{{< cite "ALLNXu0E" >}}，GraphMAE2用了该聚类方法）。

随机游走访问局部信息。使用随机游走有两大好处。一是易并行化，二是修改图结构时不需要重新运行全部算法。

#### 幂律分布（Power Low）

幂律分布的密度函数如下：

$$f(x)=ax^{-k}$$

幂律分布是长尾的，并且是尺度无关的。如果在画幂律分布时将X-Y坐标系都取log，那么密度曲线将成为直线。通过简单地推导可以得出幂律分布是唯一具有尺度无关性质的分布。

作者发现如果网络中的顶点度符合幂律分布，那么顶点出现在随机游走路径中的频率也符合幂律分布，这一现象与单词在句子中出现的频率分布相像。这启发作者将NLP中的表征方法迁移到图表征中来。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-10-04_19.55.02.png" />

#### 文本表征学习

### DeepWalk的解决方案

DeepWalk将节点看作单词，通过随机游走形成句子。接着调用文本表征方法word2vec得到节点的表征向量。

## node2vec: Scalable Feature Learning for Networks

{{< cite "BPyBK0dp" >}}提出了Node2vec，仍然基于随机游走的图表征方法。它相比于DeepWalk更加细致地设计了生成游走路线的方案。

### Node2vec的动机

随机游走是深度优先的图遍历策略，那么广度优先搜索是否可行呢？答案是肯定的。深度优先时考虑的是社区节点内部的相似性，而广度优先考虑的是邻居结构的相似性，如下图中的$u,s_6$两个节点。结构相似的两个节点可以相似很远，但它们的局部子图具有一定的相似性。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-10-04_21.37.09.png" />

### 新的游走方案

与完全随机的深度优先游走不同，作者认为图中社区内部的相似性和节点之间的结构相似性是混合存在的。因此node2vec提供了带偏好的游走方案：

当游走从节点$t$走到节点$x$，准备决定下一步时：
$$
\alpha_{pq}(t,x)=\left.
\begin{cases}
    \frac{1}{p}, & \text{if } d_{tx}=0 \\\
    1, & \text{if } d_{tx}=1 \\\
    \frac{1}{q}, & \text{if } d_{tx}=2
  \end{cases}
\right.
$$
其中$d_{tx}$表示节点$t,x$之间的最短路径长度。最终游走的概率（未归一化）为$\pi(t,x)=\alpha_{pq}(t,x)\cdot w_{vx}$，其中$w_{vx}$表示边权。

当$q>1$时，倾向于访问更近的节点，即广度优先的游走；反之当$q<1$时，倾向于访问更远的节点，即深度优先的游走。

## GCN: Semi-supervised Classification with Graph Convolutional Networks

{{< cite "14N53kyrQ" >}}提出了图卷积网络。

### 图卷积

图卷积层使用了一种”消息传递“机制，在每一层，将前一层的一阶邻居顶点的表征聚合到当前顶点。

$$
\begin{align}
H^{(l+1)} &=\sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)}) \\\
h^{(l+1)}\_i &=\sum_{j=1}^n\frac{a_{ij}}{\sqrt{\tilde{d}_i\tilde{d}_j}}h_j^{(l)} \\\
\end{align}
$$
其中$\tilde{A}=A+I_N$，$\tilde{D}\_{ii}=\sum_j\tilde{A}\_{ij}$表示度矩阵。$W\in\mathbb{R}^{n\times d}$是可训练的矩阵，它的主要作用是对表征向量降维。

### 从谱域理解图卷积

GCN的消息传递表达式容易理解，但是针对这样的设计最好要有理论解释。GCN利用图信号处理对它的设计做理论解释。

#### 图的拉普拉斯矩阵

#### 低通滤波器和图卷积

从信号处理的角度，将图的属性向量$x$的任意维度视为信号（signal）。那么，给定滤波器$g$，图卷积操作被定义为

$$x*g=Ug_{\theta}U^Tx$$
其中$g_{\theta}=diag(U^Tg)$是对角矩阵。所有的谱域GNN都采用该卷积操作的定义。不同的是对于$g_{\theta}$的选择。

- Spectral GNN将$g_{\theta}$定义为完全可训练的对角矩阵。但是由于矩阵的特征分解需要$O(n^3)$的时间，因此是不可扩展的。
- ChebNet使用Chebyshev多项式对$g_{\theta}$进行近似。

$$
\begin{align}
g_{\theta}&=\sum_{i=1}^K\theta_iT(\tilde{\Lambda}) \\\
\tilde{\Lambda}&=\frac{2\Lambda}{\Lambda_{max}}-I_n \\\
T_i(x)&=2xT_{i-1}(x)-T_{i-2}(x) \\\
T_0(x)&=1 \\\
T_1(x)&=x \\\
\end{align}
$$

其中容易推导$\tilde{\Lambda}\in[-1,1]$。基于Chebyshev多项式，图卷积操作成为

$$x*g=\sum_{i=0}^K\theta_iT_i(\tilde{L})x$$

- 现时流行的图卷积网络（Graph Convolutional Network, GCN）基于ChebNet做了进一步简化：

$$
\begin{align}
K&=1 \\\
\lambda_{max}&=2 \\\
\theta&=\theta_0=-\theta_1 \\\
\end{align}
$$

这样图卷积操作成为

$$x*g=\theta(I_n+D^{-1/2}AD^{-1/2})x$$

实际应用中，$I_n+D^{-1/2}AD^{-1/2}$会导致算术不稳定，因此GCN使用$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$进行替换。