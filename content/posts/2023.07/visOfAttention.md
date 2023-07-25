---
title: "Attention Mechanism"
date: 2023-07-10T14:47:33+08:00
draft: false
mathjax: true
bibFile: bib/bib.json
---

## Attention机制

根据OpenAI工程师[Andrej Karpathy](https://karpathy.ai/)的[讲解视频](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4185s)梳理Attention机制及其在GPT（Generative Pretrained Transformer）语言模型中的应用。在构建GPT的过程中我们会了解到attention的定义和它的工作原理。

### 构建一个小型GPT模型

GPT属于因果语言模型（Causal Language Models, CLM）。它的任务是根据当前单词（token）预测下一个单词，是自然的无监督任务。比如，现在我们有一个莎士比亚的文本数据：

```html
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

它是由字符组成的，我们需要一个映射，将其转化为模型可接受的数字向量的输入格式。首先将句子进行分词，然后建立词表，再将每个单词映射到词表的索引。这样，我们可以构建GPT的dataloader：对于给定超参数batch_size=$B$，同时给定句子片段长度$T$，dataloader可以定义为从数据中随机采样$B$个连续的长度为$T$的句子片段，来得到一个batch的数据。如下图所示。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/attn.png" width=70%/>

接着定义模型架构。这里采用经典的Transformer架构 {{< cite "3suxKdnN" >}}。如下图所示。Transformer由左侧的编码器和右侧的解码器构成，GPT采用纯解码器结构，所以这里只考虑右侧。它由N个块组成，每个块内包含了(Masked) Multi-head Attention、Add & Norm和FFN前馈网络。其中，Multi-head Attention是由多个attention块拼接起来的核心架构；Add & Norm指residual connections和layernorm，用于模型的优化；FFN是常见的全连接网络。因此，首先关注核心的attention块。

<img src="https://raw.githubusercontent.com/yliuhz/blogs/master/content/posts/images/iShot_2023-07-10_19.32.58.png" width=50%/>

### Self-Attention 自注意力机制

为了解决由前面的单词$\\{\pmb{x}\_1,\cdots,\pmb{x}\_{l-1}\\}$预测下一个单词$\pmb{x}_l$的任务，一个简单的做法是取已观测到的单词表征的平均，即

$$\pmb{x}\_l=\frac{1}{l-1}\sum\_{i=1}^{l-1}\pmb{x}\_i$$

写成矩阵即为

$$
\begin{align}
\hat{\pmb{Y}} &=\text{Softmax}(\text{Lower}(\text{ones}(T,T)))\pmb{X} \\\
&= \begin{bmatrix}
    1       & 0 & 0 & \dots & 0 \\\
    1/2       & 1/2 & 0 & \dots & 0 \\\
    \vdots & \vdots & \vdots & \ddots & \vdots  \\\
    1/T       & 1/T & 1/T & \dots & 1/T
\end{bmatrix}\_{T\times T}\times \pmb{X}\_{T\times C} \label{eq1}\tag{1}
\end{align}
$$

但这不是最优的做法，因为可能只有某几个位置的单词对待预测单词是重要的，每个单词不应授予相同的权重。因此可以考虑加权平均。注意力机制就给了一种计算权重的方法。

假设一个长度为$T$的句子的表征向量为$\pmb{X}\in\mathbb{R}^{T\times C}$。注意力机制定义了3个向量$Q,K,V$，分别表示查询向量Query，键向量Key和值向量Value。在自注意力的条件下$Q,K,V$分别由$\pmb{X}$的3个线性函数得来，即

$$
\begin{aligned}
Q &=\text{Linear}(\pmb{X}) &&=\pmb{X}\pmb{W}_Q+\pmb{b}_Q &&\in\mathbb{R}^{T\times h_s}\\\
K &=\text{Linear}(\pmb{X}) &&=\pmb{X}\pmb{W}_K+\pmb{b}_K  &&\in\mathbb{R}^{T\times h_s}\\\ 
V &=\text{Linear}(\pmb{X}) &&=\pmb{X}\pmb{W}_V+\pmb{b}_V &&\in\mathbb{R}^{T\times h_s}
\end{aligned}
$$

其中$h_s$表示输出头的维度，或称为head的维度，$\pmb{W}\in\mathbb{R}^{C\times h_s},\pmb{b}\in\mathbb{R}^{h_s}$。

每个Linear函数生成了输入$\pmb{x}$的一个代理。其中，$Q$中的每一行表示对应单词要查询的信息，$K$中每一行表示对应单词所包含的信息。这样，将$Q$的第$i$行与$K$的第$j$列做内积运算，就可以得到单词$j$是否对齐了单词$i$所要查找的信息。如果是，那么内积值会偏大，即我们想要的单词$j$对于单词$i$的权重会偏大。

因此，由$Q$和$K$计算权重矩阵，即$\text{Softmax}(\text{Lower}[QK^T])$，其中$\text{Lower}$表示取下三角矩阵，$\text{Softmax}$函数将权重规范化到$[0,1]$之间。这里$\text{Lower}$是由于在GPT的任务中，当前单词只能根据前面的单词预测，因此后面的权重是没有意义的，所以强制通过$\text{Lower}$赋成$0$。[Colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)中一个样例attention权重矩阵是

$$
\begin{aligned}
\text{Softmax}(QK^T) = 
\begin{bmatrix}
    1       & 0 & 0 & \dots & 0 \\\
    0.1574       & 0.8426 & 0 & \dots & 0 \\\
    0.2088  & 0.1646    & 0.6266 & \dots & 0 \\\
    \vdots & \vdots & \vdots & \ddots & \vdots  \\\
    0.0210       & 0.0843 & 0.0555 & \dots & 0.2391
\end{bmatrix}
\end{aligned}
$$

可以看到每个前置单词对于当前单词的权重不再相同，且每一行权重求和为$1$。

在得到权重矩阵后，将权重矩阵与值向量相乘，得到输出的词表征矩阵，即

$$\hat{\pmb{Y}} =\text{Softmax}(QK^T)V\in\mathbb{R}^{T\times h_s}$$

可以看到与上面 ($\ref{eq1}$) 式不同，自注意力机制中不是直接将权重矩阵与$\pmb{X}$相乘，而同样是用一个线性映射$V$将$\pmb{X}$包起来。[视频](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4185s)中的讲解是$\pmb{X}$相当于句子的私有特征，而$V$是$\pmb{X}$与其他位置单词交流（传播）时所使用的特征。
<!-- 这种在传播前再做一次映射的机制在图学习中也有体现，
比如在对比学习中，比较正负样本的表征时是在MLP映射后的新的映射空间做，而不是直接在GNN的输出空间做。 -->

### 关于Self Attention的一些Notes

Andrej Karpathy在一个[colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)中写了关于attention的一些notes。

- Attention机制是一种信息传递机制。它可以被看做节点从它的邻居节点通过加权平均聚合信息，而每个权重值依赖于具体的邻居节点。
- Attenion没有编码位置信息。所有聚合信息的单词对当前单词都是相同的，因此我们需要位置编码。
- Attention只在batch内的单词间进行，不同batch间的单词永远是相互独立的。
- 这里我们只考虑了纯解码器的架构。如果考虑编码器中的attention块，那么只需要把上述表达式中的$\text{Lower}$函数去掉，让单词自由地聚合信息即可。纯解码器架构采用这种半三角的权重（masking），并经常用于NLP中的自回归任务。
- Self Attention指的是$K,Q,V$由相同的输入向量$\pmb{X}$计算；反过来，Cross Attention则表示$Q$从原来的$\pmb{X}$计算，而$K,V$从其他来源计算，比如编码器的输出。而编码器-解码器的架构通常用于机器翻译任务中。编码器-解码器结构需要根据编码器的输入（如其他语言）进行输出（conditioned）。而解码器只根据前面的单词生成下面的单词（unconditioned）。
- Scaled Attention的含义是对权重矩阵做额外放缩：即乘以$1/\text{sqrt\(head_size\)}$。它可以保持权重矩阵的方差，防止在经过$\text{Softmax}$函数后退化为独热向量。这在权重**初始化**时尤其重要：如果有邻居的权重过大，那么节点只会从该邻居聚合信息，这不是我们想要的。
- Multi-head Attention（MHA）：并行地执行多个attention模块，将每个head的结果拼接作为最终输出。MHA可以提高Transformer模型的运行效率，并将学习到的不同层面的拼接在一起，有利于提高表征质量。
- 如[视频](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4185s)中所述，self-attention是一种信息传递机制。每个节点在聚合了邻居节点的信息后，需要在预测logits之前进一步映射信息到另一个空间（原文：每个节点在互相看到彼此后，还没有来得及思考它们发现了什么），这是需要在self-attention后面连接FFN的原因。

### 深层Transformer

- Residual Connections：$\hat{\pmb{Y}}=\text{Proj}(\pmb{X})+\text{Proj}(\text{Softmax}(QK^T)V)\in\mathbb{R}^{T\times C'}$，其中$\text{Proj}$是线性映射，用于转换维度以确保能够相加。[视频](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4185s)中引用了[博客](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec)。
- LayerNorm：确保数据的每行具有$0$均值和$1$方差；与之正交的BatchNorm确保数据的每列具有$0$均值和$1$方差。

### GPT模型概览

我们已经了解了构建GPT所需的所有模块。接下来小结一下GPT的预训练流程。([代码来源](https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py#L138))

{{< highlight python >}}
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    ...
{{< /highlight >}} 

首先采样一个batch的训练数据，规格为$B\times T$，每个位置的元素表示单词在词表中的下标，训练数据的标签为输入数据在句子中向后错一位的句子片段；接着将数据输入到模型中。Nano-GPT采用查表获取单词的表征，规格为$B\times T\times C$，并为句子片段中的$T$个位置通过查表得到位置编码，规格为$T\times C$，将单词表征和位置编码求和得到输入MHA的表征。接着，表征经过MHA、LayerNorm和输出头得到预测的标签logits。NanoGPT采用交叉熵损失训练。

### 利用GPT生成文本

GPT是纯解码器模型，这意味着输入一句话，GPT能够帮我们续写成一段话。

```Python
...
# GPTLanguageModel.generate
def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
```

输入的`idx`的规格是$B\times T$，在生成时$B$需要设置成$1$。输入GPT后，返回$B\times T\times \text{vocab_size}$的logits输出，每个位置表示输入相应位置的下一个单词。对于生成来说，前$T$个单词已经在输入中存在，我们只关心$T$的下一个单词，所以取logits对最后一个单词的预测`logits[:,-1,:]`。接着，对其使用`softmax`函数归一化得到概率分布，然后从该概率分布中进行采样得到要预测的下一个单词。

[很多GPT模型](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L283)在生成函数中还会加入`temperature`参数，将截断后的`logits[:,-1,:]`除以`temperature`。由于`softmax`函数的特性，在`temperature`$>1$时，归一化后的分布会更均衡，对应的GPT的表现会更发散有创造力；当`temperature`$<1$时，归一化后的分布会更确定，采样的单词会更确定，对应的GPT的表现会更准确。([参考链接](https://www.linkedin.com/pulse/text-generation-temperature-top-p-sampling-gpt-models-selvakumar/#:~:text=Temperature%3A%20This%20parameter%20determines%20the%20creativity%20and%20diversity%20of%20the%20text%20generated%20by%20the%20GPT%20model.%20A%20higher%20temperature%20value%20(e.g.%2C%201.5)%20leads%20to%20more%20diverse%20and%20creative%20text%2C%20while%20a%20lower%20value%20(e.g.%2C%200.5)%20results%20in%20more%20focused%20and%20deterministic%20text.%C2%A0))
类似地，还可以加入`topp`参数，将`logits[:,-1,:]`中概率小于`topp`的位置修改为`-inf`，这样在`softmax`归一化之后这些位置的概率变为$0$。GPT在输出时就完全不会考虑这些候选单词。

```Python
f = lambda x: np.exp(x) / np.sum(np.exp(x)) # softmax

print(f(logits))
logits = [0.10014858, 0.22968848, 0.17318473, 0.03110688, 0.46587133]

# temperature = 5
print(f(logits/5))
logits = [0.18356056, 0.21670965, 0.20481055, 0.14528531, 0.24963393]

# temperature = 0.5
print(f(logits/0.5))
logits = [0.03227246, 0.16975432, 0.09650763, 0.00311355, 0.69835204]

# topp
logits[0] = -np.inf
print(f(logits))
logits = [0.        , 0.25525156, 0.19245925, 0.0345689 , 0.51772029]
```