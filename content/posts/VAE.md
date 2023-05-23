---
title: "VAE"
date: 2023-05-23T20:42:27+08:00
draft: false
katex: true
---




## Variational Autoencoders

原博主为[Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/)

与简单的自编码器不同，变分自编码器的表征$\mathbf{z}$是一个分布。
给定从一个数据集$\mathbf{X}=\{x_i\}_{i=1}^N$，变分自编码器的观点是$\mathbf{x}$由一个隐变量$\mathbf{z}$产生，而$\mathbf{z}$则遵循一个先验分布，通常取正态分布。
因此，变分自编码器可以由3个概率分布刻画：

- $p(\mathbf{z})$: 先验分布
- $p(x|z)$: 解码器
- $p(z|x)$: 后验分布，编码器

其中后验分布很难直接计算，因此自编码器从一个未训练过的编码器，即对后验分布的估计$q(z|x)$开始，通过优化目标函数不断逼近$q(z|x)$和$p(z|x)$的距离。

这里使用KL散度衡量两个分布的距离，即$D_{KL}(q(z|x)||p(z|x))$。注意KL散度不具有对称性，原博主[Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/)甚至指出了为什么不使用$D_{KL}(p(z|x)||q(z|x))$。

在推导KL散度的表达式时就可以得到变分自编码器的损失函数ELBO。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/iShot_2023-05-23_20.26.52.png" width=80%/>
<p align="center" style="color:grey">(图源Lilian Weng的博客：https://lilianweng.github.io/posts/2018-08-12-vae/)</p>

我们想同时极大化观测数据点$\mathbf{x}$的似然，以及真假编码器的分布差距，即最大化
$$\mathbb{E}_{\mathbf{z}\sim q(z|x)}\log p(x|z)-D_{KL}(q(z|x)||p(z))$$
左边的是重构误差取反，右边的在先验分布为正态分布时可以显式展开。


在计算重构误差时用到了重参数技巧（reparameterization trick），即把从一个带参数的编码器采样$\mathbf{z}$，转化为从一个确定的分布（如标准正态）采样一个值，再通过将采样的值与编码器的输出（均值和方差）加减乘除得到$\mathbf{z}$。这样梯度就和采样独立开来，可以反向传播了。

