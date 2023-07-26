---
title: "Recurrent Neural Networks"
date: 2023-07-26T15:58:42+08:00
draft: true
mathjax: true
bibFile: bib/bib.json
---

根据其他[博客1](https://hannibunny.github.io/mlbook/neuralnetworks/02RecurrentNeuralNetworks.html)和[博客2](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)梳理循环神经网络。

## Recurrent Neural Network (RNN)

### RNN介绍

回顾多层感知机网络，每个输入只对应一个输出。当输入的数据是一个序列，且序列中的元素与其他元素相关联时，就需要用到循环神经网络。例如，句子中的单词之间是相互关联的。

<img src="iShot_2023-07-26_16.45.15.png" />

将其展开可以得到：

<img src="iShot_2023-07-26_16.45.21.png" />

用公式表示为：

$$
\begin{aligned}
f(x) &= Wx+b \\\
h_0 &= \tanh(f(x_0)) \\\
h_t &= \tanh(f(x_t)+f(h_{t-1})) \\\
y_t &= \sigma(f(h_t)) \\\
\end{aligned}
$$

其中，$x_t,h_t,y_t$分别表示$t$时刻的输入、隐藏变量和输出。可以看到每个时间戳的隐藏态都利用了前一个隐藏态。尽管展开后模型图看起来更直观，但需要注意不同时间戳$t$之间的权重矩阵（$W$）和偏移（$b$）是**共享**的。

```Python
f = lambda model: sum(p.numel() for p in model.parameters()) # 统计model的参数量

from torch import nn
rnn = nn.RNN(10, 20, 1) # 1层RNN
print(f(rnn)) # 640 = 10*20 + 20 + 20*20 + 20

rnn = nn.RNN(10, 20, 2) # 2层RNN
print(f(rnn)) # 1480 = (10*20 + 20 + 20*20 + 20) + (20*20 + 20 + 20*20 + 20)
```

当叠加RNN层时，前一层的隐藏态$h_t$就作为当前层的输入$x_t$。

### RNN的不足

