---
title: "GraphTransformer"
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