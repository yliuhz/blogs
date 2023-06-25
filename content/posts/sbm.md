---
title: "Stochastic Blockmodels"
date: 2023-06-21T15:58:20+08:00
draft: false
mathjax: true
bibFile: bib/bib.json
---

本文基于 {{< cite "1AQ92Btmc" >}} 介绍社区发现中经典的随机块模型。
随机块模型是一种生成模型，它建模了社区与图生成之间的联系。尽管简单，随机块模型可以生成非常多样的图结构，包括同配图和异配图。

## 经典的随机块模型

考虑一个无向的多重图$G$，其中两个顶点之间的连边个数可以超过$1$。令$A_{ij}$表示图的邻接矩阵，当$i\neq j$时$A_{ij}$表示顶点$i$和$j$之间的连边个数，$i=j$时表示自环个数的2倍。
随机块模型假设不同边之间符合**独立同泊松分布**($P(x=k)=\frac{\lambda^k}{k!}\exp(-\lambda)$，期望为$\lambda$)。令$w_{rs}$表示社区$r$内的**顶点**和$s$内的**顶点**之间的期望连边个数，$\frac{1}{2}w_{rr}$表示社区$r$内部顶点之间的期望连边个数。令$g_i$表示顶点$i$的社区标签。拥有上述符号和独立性假设，我们可以写出图$G$的似然函数，即将所有边的存在概率相乘:

$$P(G|w,g)=\prod_{i<j}\frac{(w_{g_ig_j})^{A_{ij}}}{A_{ij}!}\exp(-w_{g_ig_j})\times \prod_i\frac{(\frac{1}{2}w_{g_ig_i})^{A_{ii}/2}}{(A_{ii}/2)!}\exp\left(-\frac{1}{2}w_{g_ig_i}\right)$$

做合并可以得到等价的表达：

$$P(G|w,g)=\frac{1}{\prod_{i<j}A_{ij}!\prod_i2^{A_{ii}/2}(A_{ii}/2)!}\prod_{rs}w_{rs}^{m_{rs}/2}\exp\left(-\frac{1}{2}n_rn_sw_{rs}\right)$$

其中$n_r$表示社区$r$内的顶点个数，$m_{rs}=\sum_{ij}A_{ij}\delta_{g_i,r}\delta_{g_j,s}$表示社区$r$和$s$之间的合计边个数，或者在$r=s$时等于该数值的二倍。

给定观测到的图结构，我们希望$w_{rs}$和$g_i$能够最大化这个似然函数。对似然函数取对数，并忽略掉与$w_{rs}$和$g_i$无关的常数(即前面带有$A_{ij}$的分式)，得到：

$$\log P(G|w,g)=\sum_{rs}(m_{rs}\log w_{rs}-n_rn_sw_{rs})$$

首先对$w_{rs}$求导，$\frac{\partial \log P}{\partial w_{rs}}=\sum_{rs}\left(\frac{m_{rs}}{w_{rs}}-n_rn_s\right)=0$，得到$w$的最优解：

$$\hat{w}\_{rs}=\frac{m_{rs}}{n_rn_s},\forall r,s$$

将$\hat{w}\_{rs}$带回对数似然，得到$\log P(G|\hat{w},g)=\sum_{rs}m_{rs}\log(m_{rs}/n_rn_s)-2m$，其中$m=\frac{1}{2}\sum_{rs}m_{rs}$表示图中所有连边的个数，与$g_i$无关可以丢掉，因此可以得到最终的对数似然优化目标：

$$\mathcal{L}(G|g)=\sum_{rs}m_{rs}\log\frac{m_{rs}}{n_rn_s}$$

所以，随机块模型定义了一个对数似然。
接下来，我们可以使用各种方法从$K^N$空间中采样$g$，并取使得$\mathcal{L}$最大的$g$作为社区发现的输出。

## 采样方法（社区发现方法）

#### 方法1

{{< cite "1AQ92Btmc" >}} 中用自然语言描述了采样$g$的方法。首先随机地将图划分为$K$个社区（注意这里假设$K$是已知的）。接下来不断地将顶点移动到另一个使得$\mathcal{L}$增长最大的社区，或者减少最少的社区。当所有顶点移动一次后，检查移动过程中的$\mathcal{L}$值，取对应最大$\mathcal{L}$的移动结果作为下一次循环的开始状态。当$\mathcal{L}$无法被增长时算法停止。作者发现使用不同的随机种子多运行几次取最佳（$\mathcal{L}$最大的结果？）能够得到最好的结果。

#### 方法2

另一篇{{< cite "H0sxwTcP">}} 在[Supplementary information](https://doi.org/10.1038/s41598-019-49580-5)中使用了另一种采样方法。初始时仍然是随机地将顶点指定到$K$个社区中的任意一个。接下来，尝试以一定的概率将顶点从社区$r$转移到社区$s$：

$$p(r\to s|t)=\frac{m_{ts}+\epsilon}{\sum_sm_{ts}+\epsilon K}$$

其中$t$表示顶点的邻居的社区标签，$m_{ts}$表示社区$t$和$s$之间的合计边个数，或者在$t=s$时等于该数值的二倍。$\epsilon>0$通常被设置成较小的数值，在$\epsilon$趋于无穷时转化概率变为$1/K$成为完全随机地转移。对于每次尝试转移，接受该转移的概率$a$定义为：

$$a=\min \left\\{\exp(\beta\Delta\mathcal{L})\frac{\sum_tn_tp(s\to r|t)}{\sum_tn_tp(r\to s|t)},1\right\\}$$

其中转移完成后该顶点有$n_t$个邻居属于社区$t$，$\Delta\mathcal{L}$表示每次转移后对数似然的变化，$p(s\to r|t)$在顶点进行$r\to s$的转移后进行计算。$\beta$是温度参数，可以防止陷入局部最优（[Supplementary information](https://doi.org/10.1038/s41598-019-49580-5)中的式子疑似有误，比如$\min$函数里没有$1$，没有温度参数，根据{{< cite "1P6Rfctx" >}}进行了修改）。

## 随机块模型与模块度的联系

随机块模型的优化目标是对数似然，而一些热门的社区发现算法，如Louvain方法等，采用模块度作为优化目标。对数似然和模块度实际上存在联系。

将随机块模型的对数似然式子等价变换，得到

$$\mathcal{L}(G|g)=\sum_{rs}\frac{m_{rs}}{2m}\log\frac{m_{rs}/2m}{n_rn_s/n^2}$$

上式中仅添加了顶点个数$n$和边个数$m$，对于优化来说没有影响。式中有两个概率表达式：$m_{rs}/2m$和$n_rn_s/n^2$。其中，$m_{rs}/2m$表示随机采样一条边落在社区$r$和$s$之间的概率；$n_rn_s/n^2$可以看作当图中有$n^2/2$条边时的这个概率。将上述两个概率分布分别记作$p_K(r,s)$和$p_1(r,s)$，则对数似然可以被重写为

$$\mathcal{L}(G|g)=\sum_{rs}p_K(r,s)\log\frac{p_K(r,s)}{p_1(r,s)}$$

成为$p_K$和$p_1$的KL-散度。

这种定义一个空模型（null model）$p_1$的思想在模块度中也有体现：

$$Q(g)=\frac{1}{2m}\sum_{ij}[A_{ij}-P_{ij}]\delta(g_i,g_j)$$

其中$A_{ij}$表示邻接矩阵在$i,j$处的值，$P_{ij}$表示在空模型下这个值的期望。如果在模块度中使用对数似然的空模型$p_1$，那么模块度的表达式转化为：

$$Q(g)=\sum_{r=1}^K[p_K(r,r)-p_1(r,r)]$$

然而，$p_1$的最大问题是不符合实际观测到的图结构，它没有保护观测图中的顶点的度，因此实际应用中通常不会使用。如果能够保证空模型中顶点度的期望与观测图相同，那么新的空模型可以表示为

$$p_{degree}(r,s)=\frac{\kappa_r}{2m}\frac{\kappa_s}{2m}$$

其中$\kappa_r=\sum_sm_{rs}=\sum_ik_i\delta_{g_i,r}$表示社区$r$中所有顶点的度之和。（也可以从另一角度理解$\kappa_r$：将原图中所有的边切成两半准备重新随机连接，那么$\kappa_r$表示从社区$r$中伸出的半边（stub）的个数。）这样新的模块度转化为：

$$Q(g)=\sum_{r=1}^K[p_K(r,r)-p_{degree}(r,r)]$$

新的空模型$p_{degree}$导出的模块度优化目标的效果更好。这是因为具有高度数的顶点本就应有更高的概率相连，因为它们能伸出更多的半边。

受模块度的启发，作者 {{< cite "1AQ92Btmc" >}} 提出将顶点度的异构性集成到对数似然中，能得到更好的随机块模型。

## 度保护随机块模型

在经典的随机块模型中，使用$w_{g_ig_j}$刻画社区$g_i$和$g_j$之间边的期望个数，没有对社区$g_i$和$g_j$内的顶点进行区分。在新的度保护随机块模型中，引入新的变量$\theta$，其中$\theta_i$控制顶点$i$的平均度数，这样对一个观测到的图$G$，新的似然函数为：

$$P(G|\theta,w,g)=\prod_{i<j}\frac{(\theta_i\theta_jw_{g_ig_j})^{A_{ij}}}{A_{ij}!}\exp(-\theta_i\theta_jw_{g_ig_j})\times \prod_i\frac{(\frac{1}{2}\theta_i^2w_{g_ig_i})^{A_{ii}/2}}{(A_{ii}/2)!}\exp\left(-\frac{1}{2}\theta_i^2w_{g_ig_i}\right)$$

其中$\theta_i$作了归一化：$\sum_i\theta_i\delta_{g_i,r}=1$。这样$\theta_i$不再表示顶点$i$度数的期望，而是表示当一条边连向社区$g_i$时，具体连向这个顶点$i$的概率。与经典随机块模型类似，对似然函数进行等价重写：

$$P(G|\theta,w,g)=\frac{1}{\prod_{i<j}A_{ij}!\prod_i2^{A_{ii}/2}(A_{ii}/2)!}\prod_i\theta_i^{k_i}\prod_{rs}w_{rs}^{m_{rs}/2}\exp\left(-\frac{1}{2}w_{rs}\right)$$

其中$k_i$表示顶点$i$的度数。由于$\theta_i$作了归一化，$\exp$函数内不再包含$n_rn_s$。对似然函数取对数，并忽略掉与变量无关的常数：

$$\log P(G|\theta,w,g)=2\sum_ik_i\log\theta_i+\sum_{rs}(m_{rs}\log w_{rs}-w_{rs})$$

首先观察到对数似然关于$\theta_i$是递增函数，而$\theta_i$有归一化限制，且$\theta_i>0$，因此$\theta_i$的极值点为

$$\hat{\theta}\_i=\frac{k_i}{\kappa_{g_i}}$$

然后对$w$求偏导，得到$w_{rs}$的极值点为

$$\hat{w}\_{rs}=m_{rs}$$

将$\hat{\theta_i}$和$\hat{w}_{rs}$代入对数似然中，得到

$$\log P(G|\theta,w,g)=2\sum_ik_i\log\frac{k_i}{\kappa_{g_i}}+\sum_{rs}m_{rs}\log m_{rs}-2m$$

作者利用$\kappa_r=\sum_sm_{rs}=\sum_ik_i\delta_{g_i,r}$对第一项做了巧妙的等价变换：

$$\begin{aligned}
2\sum_ik_i\log\frac{k_i}{\kappa_{g_i}} 
&= 2\sum_ik_i\log k_i-2\sum_ik_i\log \kappa_{g_i}  \\\ 
&= 2\sum_ik_i\log k_i-2\sum_i\sum_rk_i\delta_{g_i,r}\log \kappa_r \\\
&= 2\sum_ik_i\log k_i-2\sum_r\sum_ik_i\delta_{g_i,r}\log \kappa_r \\\
&= 2\sum_ik_i\log k_i-2\sum_r\kappa_r\log\kappa_r \\\
&= 2\sum_ik_i\log k_i-(\sum_r\kappa_r\log\kappa_r+\sum_s\kappa_s\log\kappa_s) \\\
&= 2\sum_ik_i\log k_i-(\sum_{rs}m_{rs}\log\kappa_r+\sum_{sr}m_{rs}\log\kappa_s) \\\
&= 2\sum_ik_i\log k_i-\sum_{rs}m_{rs}\log\kappa_r\kappa_s
\end{aligned}$$

代入对数似然中，并丢弃与$g$无关的常数（$k_i,m$），得到：

$$\mathcal{L}(G|g)=\sum_{rs}m_{rs}\log\frac{m_{rs}}{\kappa_r\kappa_s}$$

可以看到，经过推导新的对数似然与经典的对数似然只有很小的差别。但是新的对数似然已经转化为$p_K$和$p_{degree}$的KL-散度，即

$$\mathcal{L}(G|g)=\sum_{rs}\frac{m_{rs}}{2m}\log\frac{m_{rs}/2m}{(\kappa_r/2m)(\kappa_s/2m)}$$

换句话说，新的对数似然找到的是与给定顶点度的随机图差别最大的社区划分，而经典的对数似然找到的是与完全随机图差别最大的社区划分。