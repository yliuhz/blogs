---
title: "Beyond Message-Passing GNNs"
date: 2023-07-18T20:07:55+08:00
draft: true
mathjax: true
bibFile: bib/bib.json
---

## Representing Long-Range Context for Graph Neural Networks with Global Attention

{{< cite "Lw8wOTy5" >}} 提出了GraphTrans，一种使用Transformer拓宽GNN感受野的方法。

### 前言：GNN的不足

GNN可以通过叠加更多的层数提高其聚合信息的邻居范围。然而，在层数不断加深时，GNN的性能也会随之下降，尤其是在全图分类和回归任务上。例如，假设一个顶点$A$需要从距其$K$-跳的顶点$B$聚合信息，那么我们可以通过叠加$K$层GNN实现。然而，GNN的感受野随着层数增加会呈指数级增长，这同时也稀释了目标顶点$B$的信息。当GNN层数过深时，图中所有顶点的表征趋向于一致，这种现象被称作过平滑现象（over-smoothing或over-squashing）。因此，人们通常会限定叠加GNN的层数。

受启发于计算机视觉中CNN的成功实践，现有方法通过添加中间的池化操作缓解GNN的过平滑。图池化通过将邻居顶点压缩为单个顶点，逐渐使顶点的表征“粗糙化”。然而，现有方法并没有找到通用有效的池化方法，且获得最优效果的SOTA方法是没有添加中间池化层的方法。

### 相关工作：图分类中的池化操作

本文专注于整图分类任务。GNN会将图的结构信息编码为顶点的表征向量，如何将顶点的表征向量整合为一个图的表征是富有挑战性的问题，也称为池化操作。池化操作有两种方式，即全局池化和局部池化：

- 全局池化：将顶点表征/边表征的集合缩小为单个的图表征；
- 局部池化：压缩顶点子集，构建新的粗糙的图结构。然而，有研究者指出使用这种池化的动机和有效性并不明确。

现在最常使用的池化操作有两种：（1）顶点表征的求和/平均；（2）添加“虚拟顶点”，与其他所有顶点相连，GNN最终输出虚拟顶点的表征作为池化后的表征。

### 动机：为何要长距离聚合信息

在计算机视觉中，研究者发现在CNN骨架网络后添加使用attention的模块取得了SOTA的效果，结果说明尽管类似于图中的关联关系对学习局部、短距离的表征很有帮助，在考虑长距离依赖时少使用结构信息反而效果更好。因此，本文的GraphTrans借助这一直觉，选择在GNN骨架后添加Transformer学习长距离依赖，这里的Transformer不使用图的结构信息：GraphTrans的Transformer使得每个顶点和其他**所有顶点**传递信息，不再局限于输入的图结构中的邻居顶点。GraphTrans期待的是前面的GNN骨架网络已经利用图结构学习了足够的邻居信息。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-07-18_20.50.10.png" />

作者做了一个实验验证和所有顶点传递信息的重要性。使用OGB Code2数据集的一个子图，任务是通过Python函数预测该函数的名字。图(b)可视化了GraphTrans中的Transformer的attention矩阵，结果显示顶点$8$从顶点$17$接收了许多信息，尽管它们在图中的距离是$5$跳。图(b)中的$18$号顶点是特殊的`<CLS>`顶点，它的表征就作为池化的结果，图(b)显示它的表征接收了大部分顶点的信息。这个实验结果表明GraphTrans是按照预想的方式工作的。

### 解决方案：GraphTrans

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-07-18_20.58.28.png" />

GraphTrans包含两个子模块：GNN和Transformer。

**GNN模块**：$\pmb{h}_v^l=f_l(\pmb{h}_u^{l-1},\\{\pmb{h}_u^{l-1}|u\in\mathcal{N}(v)\\}),l=1,\cdots,L\_{GNN}$。可以使用任意类型的GNN作为骨架。

**Tranformer模块**：得到GNN的输出$\pmb{h}_v^{GNN}$后，先将其映射到Transformer的输入维度，加一层LayerNorm。这里不加位置编码，因为假设GNN骨架已经学习了图的结构信息。之后便输入到标准的多头Transformer结构中。与GPT不同，这里的attention矩阵是全部有效的，即每个顶点可以从其他所有顶点聚合信息。

**`CLS`读出表征**：在NLP的文本分类任务中，一个通常的实践是在输入到神经网络之前，在文本的末尾添加特殊的`CLS`token，在训练时计算`CLS`和其他所有token的关系，并聚合表征到`CLS`上。在输出时，`CLS`的表征就作为整个文本的表征。

GraphTrans采用类似的操作。在输入到Transformer之前，添加一个特殊的可学习的$\pmb{h}_{\<CLS\>}$。

## Pure Transformers are Powerful Graph Learners

{{< cite "N08DsO0I" >}} 提出了TokenGT，将图中的顶点和边都看成单词（token），使用原始transformer进行学习；并从理论上证明，在一定条件下TokenGT至少具有不变图网络（invariant graph network，2-IGN）的表达能力，这已经比所有的信息传递GNN有更强的表达能力。

### 研究动机

由于Transformer的巨大成功，许多工作尝试将self-attention机制加入图学习中。由于Transformer的全局注意力无法使用给定的图结构信息，现有工作对注意力的结构进行如下三种修改：

- 使用局部注意力，即学习每条边上信息传递的权重，如GAT等；
- 在信息传递GNN后连接Transformer；
- 通过注意力偏置在全局注意力中加入图的连边信息，如Graphormer等。

然而，这些修改会限制Transformer在多模态或多任务场景的通用性{{< cite "IHZXfRPy" >}}。而且可能额外引入GNN的局限性，如过平滑等。此外，针对Transformer的工程优化方法可能无法使用，如线性注意力{{< cite "18JvSI3xc" >}}等。

因此，本文将标准Transformer直接应用于图数据。将顶点和边看作相互独立的单词，选用恰当的单词表征进行增强，同时在理论和实验上取得不错的结果。

### 解决方案：TokenGT

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-07_19.56.43.png" />

给定图$G=(V,E)$，其中$V=\\{v_1,\cdots,v_n\\}$和$E=\\{e_1,\cdots,e_m\\}$分别表示顶点集和边集。每个顶点和边分别带有顶点特征$X^V\in\mathbb{R}^C$和边特征$X^E\in\mathbb{R}^C$。因此，一共有$X=[X^V;X^E]\in\mathbb{R}^{(n+m)\times C}$的输入特征。一种naive的做法是直接将$X$输入Transformer中，但会忽视图结构以及对顶点和边的区分信息。因此，TokenGT对$X$进行两项增强：

- **顶点标识符**，用于表示（顶点和边的）连通性；
- （可训练的）**类型标识符**，用于区分顶点和边。

#### 顶点标识符

对于图$G=(V,E)$，构建正交向量集$P\in\mathbb{R}^{n\times d_p}$，称作顶点标识符。接着，对$X$进行增强：

- 对每个顶点$v\in V$的特征，将$X_v$增强为$[X_v,P_v,P_v]$；
- 对每条边$(u,v)\in E$的特征，将特征$X_{(u,v)}$增强为$[X_{(u,v)},P_u,P_v]$。

由于$P$中的向量是正交的，通过一个简单的点积操作就可以判断顶点和边是否连通。例如，想要判断边$e=(u,v)$是否与顶点$k$连通，即是否有$k\in\\{u,v\\}$，只需要检查$[P_u,P_v][P_k,P_k]^T$，它的值为$1$当且仅当$k\in\\{u,v\\}$。

由于$P$只要求正交性，有许多种生成正交向量集的方法可供选择。作者展示了两种方法：

- ORFs：对随机高斯矩阵$\pmb{G}\in\mathbb{R}^{n\times n}$，计算它的QR分解，得到$Q\in\mathbb{R}^{n\times n}$，取$Q$的每一行即可；
- 取图的Laplace矩阵的特征分解$\Delta=I-D^{-1/2}AD^{-1/2}=U^T\Lambda U$中的$U$矩阵。

#### 类型标识符

类型标识符是可训练的。对于图$G=(V,E)$，构建可训练的矩阵$E=[E^V,E^E]\in \mathbb{R}^{2\times d_e}$，接着，对$X$进一步增强：

- 对每个顶点$v\in V$的特征，将$[X_v,P_v,P_v]$增强为$[X_v,P_v,P_v,E^V]$；
- 对每条边$(u,v)\in E$的特征，将特征$[X_{(u,v)},P_u,P_v]$增强为$[X_{(u,v)},P_u,P_v,E^E]$。

**注意到$E$只有两行，即所有顶点共用一个类型表示符，所有边共用另一个类型标识符**。

#### 模型输入和结构

给定增强后的特征$X^{in}\in\mathbb{R}^{(n+m)\times(C+2d_p+d_e)}$，首先使用一个可训练的矩阵$w^{in}\in\mathbb{R}^{(C+2d_p+d_e)\times d}$变换维度，作为Transformer的输入。对于图级别的预测任务，参照流行的实践，额外添加一个特殊的单词`[graph]`，带有可训练的表征$X_{graph}\in\mathbb{R}^d$。这样，Transformer的输入为$Z^{(0)}=[X_{graph}；X^{in}w^{in}]\in\mathbb{R}^{(1+n+m)\times d}$。
模型采用标准的多头注意力和前馈网络交替堆叠的架构。

本文在正文部分使用PCQM4Mv2数据集做图回归任务实验，在附录补充了顶点分类任务的实验。

## Half-Hop: A graph upsampling approach for slowing down message passing

本文发表于[ICML2023](https://openreview.net/forum?id=lXczFIwQkv)。

### 研究动机

由于图的结构是多样的，设计一种鲁棒的图学习范式应对不同的任务是具有挑战性的。大多数GNN遵循图结构上的信息传递（Message Passing，MP）机制。MP在（1）异配图{{< cite "BebO8uzL" >}}，即连边的两个顶点具有不同的类别；（2）顶点的度数和连通性不断变化的图上{{< cite "hl1cJIOU" >}}具有局限性。着眼于现有的广泛MP-GNN架构，我们需要的是能够即插即用的方法提升GNN的通用性。

本文提出了一种新的数据增强方法，称为Half-Hop，直接修改输入图而非GNN的架构或者损失函数。Half-Hop是一种上采样方法，它引入边上的”慢节点“，用于减缓边上的信息传递。Half-Hop可同时应用于图学习中的半监督和自监督任务。

### 解决方案

对于一条有向边$(v_i,v_j)$，Half-Hop在中间添加一个顶点$v_k$，使得$v_i$到$v_k$是单向的，而$v_k$到$v_j$是双向的：

$$
\begin{align}
V'&=V\cup \\{v_k\\} \\\
E'&=(E\setminus \\{e_{ij}\\})\cup \\{e_{i\to k},e_{j\to k},e_{k\to j}\\} \\\
\end{align}
$$

对于单双向边的选择，作者在附录展示了实验结果，发现当前的选择在异配图上的*顶点分类*效果最优。
新添加的顶点$v_k$需要特征向量，Half-Hop采用差值法：

$$x_k=(1-\alpha)x_j+\alpha x_i$$

其中$\alpha$是超参数。

对于整张图，需要决定选取多少条边做插入慢顶点的增强。作者使用一个概率$p$。对于每个顶点$v_i$，以$p$的概率增强所有指向它的连边。这样得到的新的图结构写为$(V',E')\sim hh_{\alpha}(G;p)$。

对于自监督学习场景，Half-Hop可用来生成多视角学习的两个视角：

$$G_1\sim hh_{\alpha}(G;p_1),G_2\sim hh_{\alpha}(G;p_2)$$

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-01_20.51.59.png" />

### 理解Half-Hop

Half-Hop本身特别简单，作者花费另一个章节描述如何理解它的作用。