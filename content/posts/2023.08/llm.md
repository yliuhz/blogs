---
title: "LLM"
date: 2023-08-15T17:20:43+08:00
draft: true
mathjax: true
bibFile: bib/bib.json
---

## DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

### 研究动机 -- 数据混合比例

大模型的训练数据是多种来源混合的。例如，著名的The-pile数据集，包含了24%的网页数据、9%的维基百科数据和4%的Github数据等。{{< cite "1EWVWsRQN" >}} 研究了多来源数据的混合比例问题。对于The-pile数据集，DoReMi首先训练一个280M参数量的与下游任务是无关的模型，用于生成每种数据的比例；接着，按这种比例重新采样数据，并训练一个目标大规模模型（8B）。结果表明大模型在每个数据源上都获得性能提升，即使在做了下采样的数据源上。

现有得到数据比例的方法有启发式方法（The-pile）和基于下游任务的方法（PaLM、GLaM）。然而，启发式方法可能是次优的；基于下游任务的方法通常需要在多个任务和多种数据混合比例上训练上千个模型，并且在特定任务上有过拟合的风险。

### 问题设置

假设存在$k$个数据域（domain，如维基百科、Github等）。每个域的数据记为$D_i$。每个域$i$的权重$\alpha_i$定义了数据采样的分布$P_{\alpha}$，即

$$P_{\alpha}=\sum_{i=1}^k\alpha_i\cdot\text{Unif}(D_i)=\sum_{i=1}^k\alpha_i\cdot\frac{1}{|D_i|}\sum_{x\in D}\delta_{x}$$

其中，$\delta_x(x')=1$当且仅当$x'=x$，否则为$0$。例如，对于任意一个属于域$i$的样本$x'$，采样到它的概率为

$$P_{\alpha}(x')=\sum_{i'=1}^k\alpha_{i'}\frac{1}{D_i}=\frac{\alpha_i}{|D_i|}$$

这样，数据采样的过程可看作独立的两步：第一步以$\alpha$分布采样一个域，第二步以均匀分布从该域中采样数据点。
本文提出的DoReMi方法致力于找到最优的$\bar{\alpha}$，并利用分布$P_{\bar{\alpha}}$采样的数据训练最终的大模型。

### 解决方案 -- DoReMi

第一步，训练一个小规模参照模型（reference model）。其中，对照的训练数据域权重$\alpha_{\text{ref}}$可设置为均匀分布；

第二步，准备一个小规模代理模型（proxy model），初始化模型权重为$\theta_0$。初始化数据源权重$\alpha_0$为均匀分布。训练$T$代，每代得到一个数据源权重$\alpha_t$，最终返回$\bar{\alpha}=\frac{1}{T}\sum_t\alpha_t$。

在每个训练代$t\in T$，首先均匀采样一批数据$B=\\{x_1,\cdots,x_b\\}$作为训练数据。接着，先计算当前代理模型与对照模型的loss差异$\lambda_t$：

$$\lambda_t[i]\gets \frac{1}{\sum_{x\in B\cap D_i}|x|}\sum_{x\in B\cap D_i}\sum_{j=1}^{|x|}\max\\{l_{\theta_{t-1},j}(x)-l_{\text{ref},j}(x),0\\}$$

其中$l_{\cdot,j}(x)$表示$x$的第$j$个token的loss值。可以发现，右式中，乘法第一项对所有token取平均；第二项对所有token遍历；第三项逐token计算代理模型与参照模型的差异，并用$\max$忽略代理模型好的情况。那么，$\lambda_t[i]$表示了当前训练代在域$i$上代理模型比参照模型**差了多少**。
接着，由$\lambda_t$更新$\alpha_t$：

$$
\begin{align}
\alpha_t' &\gets \alpha_{t-1}\exp(\eta \lambda_t) \\\
\alpha_t &\gets (1-c)\frac{\alpha_t'}{\sum_{i=1}^k\alpha_t'[i]}+cu
\end{align}
$$

其中$\eta=1,c=10^{-3}$为超参数。可以看到，差异$\lambda_t[i]$越大，域权重$\alpha_t[i]$的增幅越大。这表示在后续训练大模型时希望加入更多域$i$的数据。

最后，利用上述$\min_\theta\max_{\alpha}L(\theta,\alpha)$更新代理模型的模型权重：

$$\min_\theta\max_{\alpha}L(\theta,\alpha)=\min_\theta\max_{\alpha}\sum_{i=1}^k\alpha_i\cdot\left[\frac{1}{\sum_{x\in D_i}|x|}\sum_{x\in D_i}l_{\theta}(x)-l_{\text{ref}}(x)\right]$$

其中$|x|$表示$x$包含的token个数，$l_{\theta}(x)=-\log p_{\theta}(x),l_{\text{ref}}(x)=-\log p_{\text{ref}}(x)$表示负对数似然损失函数。

需要注意的是，代理模型更新参数只更新$\theta$，$\alpha$在模型更新时是静态的。因此，在理解$L(\theta,\alpha)$时，$\max_{\alpha}$首先挑选出了使得当前代理模型最差的数据源分布$\alpha$，得到与参照模型差异最大的损失值；接着，$\min_\theta$更新$\theta$最小化这个最大的损失值。换句话说，代理模型是在优化最坏的情况。

$\sum_{i=1}^k\alpha_i\cdot\left[\frac{1}{\sum_{x\in D_i}|x|}\sum_{x\in D_i}l_{\theta}(x)-l_{\text{ref}}(x)\right]$是一个加权平均。要控制$\alpha$得到该加权平均的最大值，在$\sum_i\alpha_i=1$的条件下，$\max_{\alpha_i}\left(\sum_i\alpha_i\cdot C_i\right)=\max_i(C_i)$，这是因为

$$\max(C_i)-\sum_i\alpha_i\cdot C_i=\sum_i\left(\max(C_i)-C_i\right)\alpha_i\ge 0$$

其中$C_i$是常数，$i=1,2,\cdots,N$。
因此，$\min_\theta\max_{\alpha}L(\theta,\alpha)$先挑出了代理模型最差的数据域，并优化代理模型在这个域上的性能。

第三步，利用$\bar{\alpha}$重新采样数据集，并在其上训练大模型。

DoReMi框架可以迭代式地执行，即将前一次的$\bar{\alpha}$作为这次的参照模型的训练数据域分布$\alpha_{\text{ref}}$。

### 实验

本文采用两个数据集：The pile数据集和GLaM数据集，后者曾用于训练谷歌的PaLM模型。
模型架构使用纯解码器的Transformer模型。参照模型和代理模型默认都拥有280M的参数量。参照模型和代理模型始终拥有相同的参数量。

验证模型性能时有两种方式。第一种是使用留出验证集计算pplx；第二种是使用1-shot的下游生成任务，此时计算匹配准确度。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-15_21.35.44.png" />

DoReMi在The pile数据集上明显优于使用The pile原始数据集训练的基线模型。在GLaM数据集上与使用下游任务调整数据权重的方法性能相当。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-15_21.39.09.png" />

DoReMi在所有数据域上均获得性能提升，而不存在数据域之间的“trade off”。作者做出了解释。首先，具有高熵和低熵的数据域不需要过多的样本训练。低熵的数据很容易学习；高熵的数据的token趋于均匀分布，而初始化后的模型就能够在token后随机地输出下一个token。第二，中等熵的数据需要更多的样本进行训练。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-15_21.44.33.png" />

在GLaM数据集上，迭代的DoReMi取得与基于下游任务调整数据权重的方法相似的结果。作者发现该数据集变换数据源权重对结果的影响相对较少，可能是因为GLaM相比The pile的域划分（$8<22$）更粗糙。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-15_21.47.19.png" />

作者展示了DoReMi学习到的数据权重与原始权重的对比。注意到DoReMi在维基百科数据集上做了下采样，但仍然在该域的下游任务上取得性能提升。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-15_21.49.33.png" />

作者将参照模型、代理模型和最终的大模型的参数量统一，观察到在不同参数量规模时DoReMi均能获得性能提升。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-15_21.51.48.png" width=80%/>

在参数量一致时，代理模型和最终的大模型哪个更好呢？作者发现，大模型的表现更好，尤其在参数量较大时。代理模型使用加权的损失函数训练，而大模型使用标准的损失函数在重新采样后的数据上训练。训练损失的不同导致了两个模型性能的差异。在参数量增大时差异扩大。*作者最后假设使用重采样的代理模型可能能提升代理模型的性能，进而优化DoReMi？*

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-15_21.57.01.png" width=75%/>

作者验证了不同大小的代理模型对大模型性能的影响。结果显示当前采用的280M代理模型的提升最大，同时其他规格的代理模型相对基线方法也有一定提升。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-08-15_21.59.49.png" width=70%/>

作者将代理模型的损失函数中的$l_{\theta}(x)-l_{\text{ref}}(x)$替换为$l_{\theta}(x)$，即"hardest"，或替换为$-l_{\text{ref}}(x)$，即"easiest"，发现均不如当前的设定。



