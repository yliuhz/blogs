---
title: "Trajactory-User Linking Problem"
date: 2023-09-12T20:02:56+08:00
draft: true
mathjax: true
bibFile: bib/bib.json
---

## S2TUL: A Semi-Supervised Framework for Trajectory-User Linking

{{< cite "74v1AdGk" >}} 提出了轨迹-用户匹配问题的半监督方法。

### 数据集：Foursquare

作者给出了预处理好的一个数据集。

```text
train_trajs.txt: 每一行[User_id, loc_1, loc_2, ..., loc_l]，其中第i个location用一个数字loc_i表示
val_trajs.txt
test_trajs.txt

train_trajs_time.txt: 每一行[t_1, t_2, ..., t_l]，每个t_i为时间戳
val_trajs_time.txt
test_trajs_time.txt

vidx_to_latlon.txt: loc_i到经纬度的映射，用于计算实际的点的距离

vocabs_dict.txt
```

### 构图

#### Repeatability Graph

> 有相交的两条轨迹之间应该连一条边

首先，直接构建$A_{tl}\in \mathbb{R}^{N_t\times N_l}$，其中$N_t,N_l$分别表示轨迹的数量和最大的loc值。
接着，Repeatability Graph构造为

$$A_R=A_{tl}A_{tl}^T\in\mathbb{R}^{N_t\times N_t}$$

$A_R$中不再含有轨迹中具体的点信息，$A_R$的每个顶点表示一条轨迹。

#### Spatial Graph

> 对于两条轨迹$T_1=((l_{11},t_{11}),(l_{12},t_{12}),\cdots,),T_2=((l_{21},t_{21}),(l_{22},t_{22}),\cdots,)$，如果存在一个loc\_i和loc\_j，使得$d(l_{1i},l_{2j})<\epsilon_s$，那么$T_1,T_2$之间应该连一条边。

构建$A_{ll}\in\mathbb{R}^{N_l\times N_l}$，其中$A_{ll}[i,j]=1$表示第$i$个loc和第$j$个loc的距离小于给定的阈值。接着，Spatial Graph构造为

$$A_S=A_{tl}\tilde{A}_{ll}A\_{tl}^T\in\mathbb{R}^{N_t\times N_t}$$
其中$\tilde{A}\_{ll}$将$A\_{ll}$的对角线值改为0。

对于$A_{tl}[i,:]$，即第$i$条轨迹的one-hot向量，$A_{ll}[:,j]$表示和loc\_j距离相近的邻居点的one-hot向量，二者内积得到$A_{tl}\tilde{A}_{ll}[i,j]$的值，表示将第$i$条轨迹的第$j$个location的值修改为loc\_j的邻居个数（如果轨迹$i$中包含loc\_j）。最后的$A\_{tl}^T$用于匹配，即仍然要根据两条轨迹的匹配程度连边和加边权。

#### Spatial-Temporal Graph

> 对于两条轨迹$T_1=((l_{11},t_{11}),(l_{12},t_{12}),\cdots,),T_2=((l_{21},t_{21}),(l_{22},t_{22}),\cdots,)$，如果存在一个loc\_i，使得$d(l_{1i},l_{2j})<\epsilon_s$**同时**$d(t_{1i},t_{2j})<\epsilon_t$，那么$T_1,T_2$之间应该连一条边。

对于Spatial Graph中的每一条边，验证它们是否同时满足时间戳的相近性。需要注意由于定义轨迹的邻近性时都使用了`存在`性定义，而ST邻近性要求空间邻近和时间邻近在一个点同时满足，因此对于空间邻近的两条轨迹，需要重新验证每一对点之间是否满足空间邻近性。若满足空间邻近性，才能继续考察时间邻近性。

#### 组合图信息

将Spatial Graph和Spatial-Temporal Graph分别与Repeatability Graph组合，并使用不同的`edge_type`区分，得到异构图。

### 模型

#### 图表征模块

依据输入图是否异构选择`GCN`或`RGCN`。

#### LSTM模块

构造的图中的顶点都没有初始表征，因此作者考虑用LSTM得到每条轨迹的初始表征。

#### 分类器

分类器使用简单的前馈网络。损失函数使用交叉熵。

$$\mathcal{L}=-\sum_{i=1}^{N_t}\sum_{k=1}^{N_u}y_i^t[k]\log y_i[k]$$

其中带$t$表示是ground_truth的one-hot向量。

#### 推理时的小细节

由于一个人在同一时间不能处于两个位置，因此如果两条轨迹都想连接到同一个用户，且轨迹在同一时间戳有两个不同的位置点，那么就只能选择其中一条轨迹连接到该用户。

作者将测试集所有轨迹对所有用户的预测结果根据预测分数进行排序。先将分数高的轨迹连接到用户，后面遇到冲突时直接跳过，直到找到每条轨迹对应的用户。

### 实验

本文将TUL问题建模为多分类问题。评估参量有

- $ACC@K_a$：如果真实用户出现在排好序的前$K_a$个分数中，$ACC@K_a=1$
- Macro-P,Macro-R,Macro-F1：分别表示每个用户上的precision,recall,F1的平均值。例如，

$$\text{Macro-F1}=\frac{\sum_{i=0}^{N_u}F1_i}{N_u}$$

实验结果显示使用LSTM并不会一直得到最优的预测准确度。