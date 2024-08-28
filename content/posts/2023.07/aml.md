---
title: "Anti-Money Laundering"
date: 2023-07-26T10:34:40+08:00
draft: true
mathjax: true
bibFile: bib/bib.json
---

反洗钱是异常检测的一个子问题。

## Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics

{{< cite "YFznIUBN" >}} 贡献了当时最大的反洗钱图数据集[Elliptic](https://www.kaggle.com/ellipticco/elliptic-data-set)，并对比了反洗钱领域常用的方法的效果，如逻辑斯蒂回归，随机森林，图卷积网络等。基于该数据集在后续工作 {{< cite "4QBCTyMI" >}} 中提出了EvolveGCN，一种可应用于动态图的图卷积网络。由于对金融名词术语不熟悉，一部分会用GPT翻译的中文结果书写并在文中标出。

### 研究动机

> It’s expensive to be poor. 

作者首先指出，限制性准入（例如开立银行账户的能力）的问题，在一定程度上是反洗钱(AML)法规日益严格的意外后果，尽管这些法规对于保护金融体系至关重要，但对低收入人群、移民和难民产生了不成比例的负面影响。约有17亿成年人无银行账户。相对高成本的问题也在一定程度上是AML政策的结果，这些政策对汇款服务企业施加了高额的合规固定成本以及由于不合规而受到刑事和货币处罚的恐惧——“低价值”客户根本不值得冒险。

然而，不能简单地认为反洗钱法规过于繁重。毒品、人口贩运和恐怖组织等数十亿美元的非法产业在全球范围内造成了巨大的人道主义灾难。最近的1MDB洗钱丑闻侵吞了马来西亚人民超过110亿美元的纳税资金，这本应该用于国家发展，高盛等其他牵涉其中的机构因此受到了巨额罚款和刑事指控。更近期的丹斯克银行洗钱丑闻发生在爱沙尼亚，作为来自俄罗斯和阿塞拜疆约2000亿美元非法资金流动的中心枢纽，同样对这些国家的无辜公民造成了难以估量的损失，丹斯克银行和德意志银行等牵涉其中的机构遭受了数十亿美元的损失。

洗钱不是一种无害的犯罪，现有的传统金融系统很难制止洗钱。**在不将这个复杂的挑战简化为仅仅数据分析的前提下，我们提出一个问题：借助正确的工具和公开数据，我们能否帮助调和安全需求和金融包容的原因?**（研究动机节由GPT翻译）

### Ellipic数据集

Ellipic是一家保护加密货币免受犯罪侵害的公司。Ellipic图数据集从原始比特币数据（438M个顶点，1.1B条边）中提取，一共包含203,769个顶点和234,355条有向边。其中，顶点表示事务，边表示钱款流向。2\%（4,545）个顶点的标签为非法，21\%（42,019）个顶点的标签为合法，其他77\%的顶点无标签。每个顶点附带166维的特征向量，前94维表示事务的局部信息，包括时间戳，输入/输出的数量，事务费用，出货量等；剩余的72维特征由该顶点的1阶邻居顶点的部分特征得来，如1阶邻居顶点的事务费用的最大值、最小值、标准差、相关系数等。

除此之外，Ellipic包含49个时间戳，时间戳之间均匀地相差两周。每个时间戳对应图中的一个连通分量，即不同时间戳的图之间没有连边。每个时间戳中的顶点个数及合法事务与非法事务的比例如下图所示。可以看到标签分布是极不均衡的。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-07-26_15.21.55.png" />

对Ellipic顶点打标签的过程是一个**启发式**的过程。比如，输入次数更多以及重复使用同一地址通常与更高的地址聚类相关，这会降低签署交易的实体的匿名性。另一方面，在一笔交易中合并多个地址控制的资金，在交易费用方面具有优势。因此，为大量用户请求放弃匿名保护措施的实体可能是合法的(例如交易所)。相反，非法活动可能倾向于选择输入次数较少的交易，以减少匿名地址聚类技术的影响。（该段由GPT翻译）

**任务是预测每个顶点是否合法的二分类问题。**

### 基准方法

常用的基准方法包括逻辑斯蒂回归，随机森林，多层感知机等。其中，逻辑斯蒂回归具有良好的可解释性，随机森林具有更高的准确率。但这类方法不使用图结构信息。为了缓解这个问题，Ellipic数据集在前94维顶点特征的基础上，添加了一阶邻居的72维信息。
除此之外，GCN等图神经网络能够直接在图结构上传播特征，也是不错的选择。

在GCN的基础上，作者提出动态图神经网络EvolveGCN，将GCN与RNN结合。EvolveGCN对每个时间戳训练一个GCN，并使用序列神经网络将不同时间戳GCN的权重联系起来。

### 实验

作者发现，在顶点特征中加入一阶邻居信息，同时加入GCN得到的表征之后，随机森林能够取得最优的效果。如下表所示，其中AF和LF分别表示使用全部的166维特征和前94维特征，NE表示进一步拼接GCN输出的表征。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-07-26_20.56.57.png" width=75%/>

同时，所有方法在面对“黑市关闭”（The Dark Market Shutdown）时都会失效。如下图所示，在t=43时黑市突然关闭，此时所有方法都会失效。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-07-26_20.57.07.png" />

## Diga: Guided Diffusion Model for Graph Recovery in Anti-Money Laundering

{{< cite "5aIS0RLz" >}} 研究了半监督洗钱检测的方法。首先使用PageRank采样子图，接着利用CV领域的扩散（diffusion）过程，将子图加噪声后重建的图与原图进行比较，从而得到该子图中心节点是否异常的分数。

### Diga -- 解决方案

#### Personalized PageRank (PPR) -- 采样中心节点的子图

PageRank用于评估每个节点对于中心节点的重要性。计算好重要性分数后，取前$M$个最重要的节点构成中心节点的子图。

作者将{{< cite "1NwIeJDz" >}}的1-跳PPR扩展为$K$-跳PPR。另外，作者发现一些特殊人群，如商人、零售业老板、优步司机等，有很大的顶点度数（即很多金钱交易），但其实与洗钱行为无关，属于正常的金钱交易。因此，在PPR中使用带参数的参数$\rho(u_i)$，从而对一些特殊人群$u_i$使用专家信息作为先验，指导PPR的计算过程。

#### 扩散过程 -- 计算异常指数

在得到中心节点的子图后，作者希望使正常节点的子图重建容易，而异常节点的子图重建困难，从而达到分类的目的。
作者训练两个模型：

- 去躁网络：用于重建加了高斯噪声的子图**属性矩阵**
- 分类器：使用给定标签训练，用于指导去躁网络

去躁网络与一般的扩散模型相似，由正向的加噪声过程和反向的去躁过程构成。其中，加噪声过程为

$$q(Z_c^t|Z_c^{t-1})=\mathcal{N}(Z_c^t;\sqrt{1-\beta^t}Z_c^{t-1},\beta^tI)$$
其中$Z_c^0=X_c$为初始属性矩阵。

反过来，去躁过程中，作者使用一个GNN网络$\epsilon_{\theta}(G,Z^t,t)$估计正态分布的均值，具体如下：

$$
\begin{align}
p_{\theta}(Z_c^{t-1}|Z_c^t) &=\mathcal{N}(Z_c^{t-1};\mu_{\theta}(G_c,Z_c^t,t),\sigma_t^2I) \\\
\mu_{\theta}(G_c,Z_c,t) &=\frac{1}{\sqrt{\alpha_t}}(Z_c^t-\frac{\beta^t}{\sqrt{1-\overline{\alpha}^t}}\epsilon_{\theta}(G_c^t,Z_c^t,t)) \\\
\end{align}
$$

接着，作者训练额外的分类器，用于指导上述去躁过程。类比于图像生成中，使用不同的条件指导生成图片的风格（如卡通等）。具体来说，分类器$p(y|G_c,Z_c,t)$使用子图结构、加噪声的属性矩阵作为输入，输出子图的中心节点是否异常的分类标签。该分类器使用训练集的标签进行训练。训练完成后，指导GNN反向去躁过程生成正常的子图属性矩阵。这样，**可以通过比较生成的正常属性和原属性的差别（如L2距离）来得到原来是否异常。**

### 带有变差结果的消融实验？

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-10-10_22.04.46.png" />

本文实际上写了3个技术贡献点，分别为PPR采样、分类器指导扩散过程，以及GNN权重共享。但消融实验显示第三个贡献点会导致准确率变差？
实际上第三个贡献点在于加速模型训练，但会导致准确率变差。

## Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision

{{< cite "EdWYslIq" >}}研究半监督图异常检测问题。作者发现异常点和正常点的一节邻居标签分布相似，从而设计新的图神经网络层ConsisGNN。并且在训练时将节点mask向量加入可训练参数，从而达到自动化数据增强的目标。一致性训练属于[半监督训练的技巧](https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch)。



## Realistic Synthetic Financial Transactions for Anti-Money Laundering Models

{{< cite "Yfy2yMFF" >}} 人工合成数据用于训练反洗钱模型。作者使用合成数据的动机有3条：

- 每个银行只有用户在行内的交易数据，不能看到用户和其他银行的交易数据
- 真实数据的洗钱标签是不完整的，因为许多洗钱行为没有被发掘
- 在真实数据上做反洗钱建模需要额外的步骤

如果人工合成的数据可以充分拟合真实数据中的交易行为，那么将有如下的好处：（1）标签可以是完整的；（2）可以模拟验证假设所有银行都共享数据，那么集中构建的模型带来的性能提升有多大。可以假设每个银行只拥有一个子图。

### 有向多重图

金融交易数据可以用有向多重图建模，其中**节点**表示银行账户，**边**表示交易行为。每次交易形成一条连边，即边上带有时间戳信息，因此形成有向多重图。如下图所示。每条边代表的交易被分类为"正常"或"洗钱"交易，因此是边分类问题。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-13_10-51-47.png" />

### 洗钱的八大模式

- Fan-out: 从一个节点指向多个($\ge 2$)其他节点
- Fan-in: 从多个($\ge 2$)节点指向一个节点
- Gather-scatter: Fan-in后Fan-out
- Scatter-gather: Fan-out后Fan-in
- Simple cycle: 闭环，资金经过周转后回到起点账户
- Random: 与cycle类似，区别在资金最终没有回到起点账户
- Bipartite: 从一个节点集合指向另一个节点集合
- Stack: 两个bipartite模式拼接起来

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-13_10-45-44.png" />

在八大模式中，所有的洗钱用户实际控制了洗钱交易路径上的所有账户。例如，洗钱用户有空壳公司，名义上路径上的账户"归属"空壳公司，但实际归属于洗钱用户。

### 金融数据合成器AMLWorld

洗钱周期包含3部分：（1）不合法的收入来源，如走私；（2）将不合法资金混进金融系统；（3）支出不合法资金

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-13_12-06-29.png" />

AMLWorld考虑多类型的用户实体，如个体户和公司账户。交易行为包括发工资，交罚款，买东西等。不合法资金来源包括9个：敲诈勒索、放高利贷、赌博、卖淫、绑架、抢劫、贪污、毒品和走私。每个实体在每个时间戳通过图中的A和B生成交易数据。A/B路径中每个节点的yes/no选择基于预定义的**统计分布**。AMLWorld采用严格的方式对洗钱行为打标签。具体来说，假设账户P向账户Q支付100元不合法来源的资金，Q向W支付其中的50元，W向Y支付50元中的25元，那么路径上的所有交易即P-Q, Q-W, W-Y都被标记为"洗钱"行为，原因是资金的来源（100元）是不合法的。AMLWorld在打标签时完全考虑资金的来源。*如果交易的资金包含合法和不合法来源，那么该笔交易仍然被标记为洗钱*。

AMLWorld根据美联储2019年公开报告数据设计交易的**统计分布**，包括单个账户的交易数量，交易方式（现金、电汇和支票等）的分布，交易金额的分布。洗钱行为除了服从八大模式外，还包含大量其他伪装后的交易行为。如下表所示，全部100K的洗钱行为中，只有19K服从八大模式，其余81K交易伪装成正常交易，如公司发工资或采购物品等。本文开源了3中规模的合成数据，每种规模包含两种洗钱交易的比例（高洗钱比例/低洗钱比例）。如下表所示。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-13_12-37-35.png" />

### 现有模型效果测试

#### 实验设置

本文验证图神经网络模型和树分类模型对于洗钱交易检测的效果。

- 梯度增强树GBT：LightGBM和XGBoost。为了使GBT模型能利用图结构信息，额外使用GFP {{< cite "3D4nhcoE" >}} 生成图结构特征。GFP采用一个batch的边作为输入，将输入边插入到GFP内部的图结构中，输出GFP内部图结构的特征，如scatter-gather结构的数量，temporal cycle的数量，simple cycle的数量等。由此可见，GFP逐渐丰富内部图的结构。输入边按照时间戳进行排序。按照下述数据划分，训练集不会使用验证/测试集生成特征。
- 消息传播型图神经网络GNN：
  - GIN with edge features 
  - GIN + EU (edge_updates) {{< cite "vU3izXD5" >}}
  - PNA {{< cite "CVg73F4B" >}}

每个GNN模型包含边的读出(readout)层，它的输入是边的表征和两个端点的表征，输出边的最终表征。在训练GNN模型时，采样100个1-跳邻居和100个2-跳邻居，但仍然无法在最大的合成数据集HI-Large和LI-Large上训练。

由于洗钱标签相对于正常标签的极度不均衡性，本文使用洗钱标签的F1-Score (Minority Class F1 Score)作为评估模型的指标。

GBT的数据划分采用时间维度的训练/验证/测试数据划分。每个划分内的节点（表示账户）个数相同，区别在于连边（表示交易）。锚定两个时间点$t_1$和$t_2$，则

- $t_1$之前的交易用于训练
- $t_1$和$t_2$中间的交易用于验证
- $t_2$之后的交易用于测试

GNN的数据划分与GBT类似：

- 训练图只包含$t_1$之前的交易，全部用于训练
- 验证图包含$t_2$之前的交易，只有$t_1$和$t_2$中间的交易用于验证
- 测试图包含全部交易，只有$t_2$之后的交易用于测试

#### 实验结果

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-13_13-00-06.png" />

表2说明树模型最优。

表3使用HI数据训练，在LI上测试（或在LI上微调后测试）。HI数据集包含更多比例的洗钱交易。结果显示大多数模型相比只在LI数据上训练（表2）有较大提升。

图6模拟现实场景，每个银行只有行内的交易数据。分3种情况：（1）私有数据且私有模型；（2）私有数据但共享模型，即共享GFP的输出；（3）共享数据且共享模型。结果对应的模型是GFP+LightGBM。结果显示共享的越多模型效果越好。

## Rethinking Graph Neural Networks for Anomaly Detection

{{< cite "lCx0Rztr" >}} 发现在正常图中加入异常点后，谱能量会更多偏向特征值高的区域。基于这种观察，作者设计BWGNN，能够更灵活建模当前节点和邻居节点的关系（相似/不相似）。

## SEFraud: Graph-based Self-Explainable Fraud Detection via Interpretative Mask Learning

{{< cite "9Vh0zoEr" >}} 设计了异构图异常点检测方案SEFraud，通过同时训练对原始节点特征和边特征的mask向量实现结果的可解释性。

SEFraud将初始特征、GNN的输出embedding和节点种类编码拼接后，经过MLP预测得到节点的mask向量；将边的两个端点的embedding和边种类编码拼接后，经过MLP预测得到边的mask向量。每个种类的节点共用一个MLP，每个种类的边共用一个MLP。节点的MLP应对特征的每个维度输出一个重要性分数，因此输出是$d$维；边的MLP应对每条边输出一个重要性分数，因此输出是$1$维。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-27_20-24-56.png" />


得到节点mask向量后，与原始特征点乘得到新的节点特征；得到边mask后，得到每条边的权重。
作者认为mask向量能帮助提升检测效果。因此，将使用mask的节点标签输出作为正样本，将mask取反后的节点标签输出作为负样本，构建对比学习损失函数，与分类损失函数一起优化。

### 解释GNN的预测输出

{{< cite "FtubF2ai" >}} 给出了解释GNN模型对节点标签输出、边标签输出和图标签输出的问题定义。对于一个属性图$G=(A,X)$，解释器需要找到一个最重要的子图$G_S=(A_S,X_S)$，GNN在该子图上的预测结果和使用原图的预测结果相近。具体来说，

对于节点分类任务，对于每个目标节点，解释器的目标是找到使得互信息最大的子图$G_S$，即

$$\max_{G_S}MI(Y,(G_S,X_S))=H(Y)-H(Y|G=G_S,X=X_S)$$

直观上说，对于目标节点$v_i$，如果移除$G_S$的节点$v_j$（或边$\langle v_j,v_k\rangle$）后，$v_i$的预测结果发生很大变化，那么节点$v_j$（或边$\langle v_j,v_k\rangle$）对于$v_i$的预测至关重要。
同时，通过学习特征的mask向量，对节点特征也找到最重要的几个维度。

对于“后解释”模型，即在GNN训练好后冻结GNN权重，训练其他模型作为GNN的解释器方法，$H(Y)$对于给定的输入是确定的，因此最大化互信息等价于最小化$H(Y|G=G_S,X=X_S)$。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-28_15-41-19.png" />

在实验中，GNNExplainer在随机图上随机选定节点连边特定子图结构（motif），并将选定的节点打上motif的标签。GNN在预测选定节点的标签时，解释器应该返回motif的结构作为与未选定节点区别最大的最重要子图结构。在下图中，红色节点是目标节点（也是被选定的节点），可以看到GNNExplainer相比基线方法更准确地返回motif结构。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-28_16-04-34.png" />

## Partitioning Message Passing for Graph Fraud Detection

{{< cite "U7YTzPoR" >}} 认为GCN在消息传递时对所有邻居节点共用权重矩阵是次优的，因此利用训练集已知的节点标签将邻居划分为正常、异常和未知，将全部邻居共用权重矩阵修改为同一类邻居共享权重矩阵：$W_{be}, W_{fr}, W_{un}$。其中未知邻居的权重矩阵$W_{un}$由前两个的线性组合得到，即$W_{un}=\alpha W_{fr}+(1-\alpha)W_{be}$，这是因为未知并不代表它们属于第三类。

另外，作者可能参考GAT中attention的计算方法，使用额外的MLP对每个节点计算权重矩阵，不过都由共享的MLP计算得来.

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/Snipaste_2024-08-28_18-01-52.png" />