---
title: "Generalization of GNNs and MLPs"
date: 2023-06-26T20:08:51+08:00
draft: false
mathjax: true
bibFile: bib/bib.json
---

本文是 {{< cite "18wo6Er1H" >}} 的论文解读。OpenReview显示这篇论文是ICLR2021的Oral论文（前5%）。

## 引言

人类具有泛化性，例如学会算术后可以应用到任意大的数字。对于神经网络而言，前馈网络（也叫多层感知机，MLPs）在学习简单的多项式函数时就不能很好地泛化了。然而，基于MLP的图神经网络（GNNs）却在近期的一些任务上表现出较好的泛化性，包括预测物理系统的演进规律，学习图论算法，解决数学公式等。
粗略地分析可能会觉得神经网络可以在训练分布以外的数据上有任意不确定的表现，但是现实中的神经网络大多是用梯度下降训练的，这就导致其泛化性能有规律可以分析。作者使用"神经切线核"（neural tangent kernel，NTK）工具进行分析。

本文的第一个结论是使用梯度下降训练的MLPs会收敛到任意方向线性的函数，因此MLPs在大多数非线性任务上无法泛化。
接着本文将分析延伸到基于MLP的GNNs，得到第二个结论：

## 前置知识

设$\mathcal{X}$表示数据（向量或图）的域。任务是学习一个函数$g:\mathcal{X}\to \mathbb{R}$，其中训练数据$\\{(\mathbf{x}_i,y_i)\\}\in\mathcal{D}$，$y_i=g(\mathbf{x_i})$，$\mathcal{D}$表示训练数据的分布。在训练数据和测试数据同分布的情况下，$\mathcal{D}=\mathcal{X}$；而在评估泛化能力时，$\mathcal{D}
\subsetneq\mathcal{X}$。一个模型的泛化能力可以用**泛化误差**评估：设$f$为模型在训练数据上得到的函数，$l$为任意损失函数，则泛化误差定义为$\mathbb{E}\_{\mathbf{x}\sim \mathcal{X} \setminus \mathcal{D}}[l(f(\mathbf{x}), g(\mathbf{x}))]$

图神经网络GNNs是在MLPs基础上定义的网络。具体来说，初始顶点表征为$\mathbf{h}_u^{(0)}=\mathbf{x}_u$。在第$k=\{1..K\}$层，顶点表征更新公式为

$$\begin{aligned}\mathbf{h}\_u^{(k)}&=\sum_{v\in\mathcal{N}(u)}\text{MLP}^{(k)}(\mathbf{h}\_u^{(k-1)},\mathbf{h}\_v^{(k-1)},\mathbf{w}\_{(v,u)}) \\\ 
\mathbf{h}\_G&=\text{MLP}^{(K+1)}(\sum_{u\in G}\mathbf{h}\_u^{(K)})\end{aligned}$$

其中$\mathbf{h}\_u^{(k)}$表示第$k$层GNN输出的顶点$u$的表征，$\mathbf{h}\_G$表示整张图的表征。$\mathbf{h}\_u^{(k)}$的计算过程称为聚合，$\mathbf{h}\_G$的计算过程称为读出。以往研究大多使用求和聚合与求和读出，而本文指出替换为另外的函数能够提升泛化性能。

## 前馈网络MLPs如何泛化

作者用下图呈现MLPs的泛化方式。灰色表示MLPs要学习的函数，蓝色和黑色分别表示模型在训练集和测试集上的预测。可以看到模型可以拟合训练集上的非线性函数，但脱离训练集后迅速变为线性函数。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/iShot_2023-06-26_20.54.05.png" />

