---
title: "LLM and Graphs"
date: 2023-10-10T22:17:37+08:00
draft: true
mathjax: true
bibFile: bib/bib.json
---

大语言模型火起来后，图机器学习社区的研究者也开始探究大语言模型在图数据上的可能应用。

## Talk like a Graph: Encoding Graphs for Large Language Models

{{< cite "NfltEOJJ" >}} 研究了使用大语言模型解决简单图论问题的表现，包括了如下几个任务：

- 边查询：查询边在图中是否存在；
- 顶点度查询：计算指定顶点的度；
- 顶点个数查询：计算图中的顶点个数；
- 边个数查询：计算图中的边个数；
- 顶点连通性查询：查询与给定顶点连通的所有顶点；
- 环路查询：判断图中是否含有环路；
- 顶点不连通性查询：查询与给定顶点不连通的所有顶点。

这些任务极其简单，是更复杂问题的中间步骤。例如，要计算最短距离，首先要找到连通的顶点；要计算图中的社区，首先需要判断是否有环路；要搜索最具影响力的顶点，首先要计算顶点的度数。

本文考虑简单的图结构$G=(V,E)$，包括各种生成随机图的算法生成的图：
- Erdos-Renyi (ER) graphs
- Scale-free networks (SFN)
- Barabasi–Albert (BA) model
- stochastic block model (SBM)
- star graphs
- path graphs
- complete graphs

如下图所示。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-10-17_16.52.27.png" />

为了使用语言模型解决图论问题，第一要把图结构编码为大语言模型的token序列，即用自然语言描述图；第二要选用合适的prompt提示方法向语言模型提问图论问题。

### 图结构编码的prompt提示方法

作者首先分别考虑如何用语言描述顶点和边，接着将它们结合起来描述图结构。

作者列出了5种描述顶点的语言：(n.1) 自然数；（n.2）广泛熟知的英文名字（如 David）；（n.3）《权利的游戏》和《南方公园》电视节目中演员的名字；（n.4）美国政治家的名（first name）；(n.5) 字母。

6种描述边的语言：（e.1）括号，即(起始顶点，终止顶点)；（e.2）朋友，即两个端点是朋友；（e.3）合著者；（e.4）社交网络，即两个端点有联系；（e.5）箭头，即起始顶点$\to$终止顶点；（e.6）描述（incident），即直接描述起始顶点与终止顶点连接。

最终得到描述图结构的方法：

Adjacency：(n.1) + (e.1)

Incident: (n.1) + (e.6)

Friendship: (n.2) + (e.2)

Co-authorship: (n.2) + (e.3)

SP: (n.3，南方公园) + (e.2)

GOT: (n.3，权利的游戏) + (e.2)

Social Network: (n.2) + (e.4)

Politician: (n.4) + (e.4)

Expert: (n.5) + (e.5)，该方法开头先说”你是个图分析师“（You are a graph analyst）。

具体的描述图结构的例子可以参加论文附录A.1。

### 提问问题的prompt提示方法

作者列举并验证了5种提问时的prompt方法：

- Zero-shot prompting(Zero-shot): 直接给出问题描述；
- Few-shot prompting(Few-shot): 先给出相关的例子，再描述问题；
- Chain-of-thought(CoT): 先给出相关的例子，在例子中展示如何一步一步地解决问题，再描述问题；
- Zero-shot CoT(Zero-CoT): 先说”Let's think step by step“，再描述问题；
- Bag prompting(CoT-Bag): 特定于图问题的prompt方法。先说”Let's construct a graph with the nodes and edges first“，再描述图结构，再描述问题。

除此之外，作者额外考虑描述问题的两种方式：（1）graph question encoder: 直接提问，如（"What's the degree of node i?"）;（2）application question encoder: 使用实际场景中的词语提问，如查询顶点度的问题变成（"counting the number of friends for an individual"）。

### 实验结论

- **大语言模型在这些简单图论问题上的表现仍然较差**；
- 描述图结构的方法对效果影响很大；
- 模型的参数量对效果影响很大。