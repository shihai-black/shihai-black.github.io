---
title: Transomer学习
date: 2021-04-11 11:10:00
categories:
- 深度学习
tags:
- transfomer
description:
- 此文档写于2019年，建成于博客创立之前。
---


# Transfomer拆分

为了更好的学习当前NLP主流模型，如Bert，GPT2及Bert一系列的衍生物，Transfomer是这一系列的基础。因此本文的主要目的是记录个人基于一些博客和原论文对Transfomer模型进行拆分的结果。

**目的**：减少计算量并提高并行效率，同时不减弱最终的实验结果。

**创新点**：

1. Self-attention
2. Multi-Head Attention

# 背景知识

## seq2seq

定义：seq2seq模型是采用一系列项目(单词、字母、图像特征等)并输出另一个项目序列的模型。在机器翻译中，序列是一系列单词，经过seq2seq后，输出同样是一系列单词。

<video src="F:\video\seq2seq_2.mp4"></video>

接下来我们掀开这个model内部，该模型主要由一个Encoder和一个Decoder组成。

- Encoder：处理输入序列的每个项目，捕捉载体中的信息（context）。
- Decoder：处理完整序列后，Encoder将信息（context）传递至Decoder，并且开始逐项生产输出序列。

<video src="F:\video\seq2seq_4.mp4"></video>

而context是一个向量，其大小基于等同于编码器中RNN的隐藏神经元。

在单词输入之前，我们需要将单词转化为词向量，其可以通过word2vec等模型进行预训练训练词库，然后将单词按照词库的词向量简单提取即可，在上面的资历中，其处理过程如下：

![embedding](F:\video\embedding.png)



这里简单将其设为维度4，通常设为200或300，在此，简单展示一下RNN的实现原理。

<video src="F:\video\RNN_1.mp4"></video>

利用先前的输入的隐藏状态，RNN将其输出至一个新的隐藏状态，接下来我们看看seq2seq中的隐藏状态是怎么进行的。

<video src="F:\video\seq2seq_5.mp4"></video>

Encoder中最后一个hidden-state实际就是前文提到的context，接下来我们进一步拆解，展示其具体细节。

<video src="F:\video\seq2seq_6.mp4"></video>

总结：由于在Encoder阶段，每个单词输入都会产生一个新的hidden_state，最后输出一个context给Decoder进行解码。因此，当文本内容过长时，容易丢失部分信息，为了解决这个问题，Attention应运而生。

## Attention

Attention这个概念最早出现在《Neural machine traslation by jointly learning to align and translate》论文中，其后《Neural image caption generation with visual attention》对attention形式进行总结。

定义：为了解决文本过长信息丢失的问题，相较于seq2seq模型，attention最大的区别就是他不在要求把所以信息都编入最后的隐藏状态中，而是可以在编码过程中，对每一个隐藏状态进行保留，最后在解码的过程中，每一步都会选择性的从编码的隐藏状态中选一个和当前状态最接近的子集进行处理，这样在产生每一个输出时就能够充分利用输入序列携程的信息，下面很好的展示了attention在seq2seq模型中运用。

<video src="F:\video\seq2seq_7.mp4"></video>

接下来，我们放大一下decoder对于attention后隐藏状态的具体使用，其在每一步解码均要进行。

1. 查看在attention机制中每个具体的隐藏状态，选出其与句子中的那个单词最相关。
2. 给每个隐藏状态打分。
3. 将每个隐藏状态进行softmax以放大具有高分数的隐藏状态。
4. 进行总和形成attention输入Decoder的向量。

<video src="F:\video\attention_process.mp4"></video>

最后，将整个过程放一起，以便更好的理解attention机制(代码实现的时候进一步理解)。

1. Decoder输入：初始化一个解码器隐藏状态+经过预训练后的词向量。
2. 将前两者输入到RNN中，产生一个新的隐藏状态h4和输出，将输出丢弃(seq2seq是直接将context送入下一个RNN作为输入)。
3. attention：利用h4和encoder层简历的隐藏状态进行计算context（c4）。
4. 将c4和h4进行concatenate。
5. 将其结果通过前向神经网络，输出一个结果。
6. 该结果表示当前时间输出的对应文字。
7. 重复下一个步骤。

<video src="F:\video\attention_tensor_dance.mp4"></video>

注意：该模型并不是将输入和输出的单词一一对应，他有可能一个单词对应两个单词甚至影响第三个单词，这是经过训练得到的。

# 正文

总算要写到Transformer部分了，有点小激动，让我们一起来看看这个影响到现在的模型到底长啥样，为了便于理解，我这边会结合代码+论文进行讲解。

首先，引入原论文的结构图

![结构图](F:\video\结构图.png)

看不懂不要紧，这边引入代码的结构图

![代码结构图](F:\video\代码结构图.png)

## Encoder-Decoder

从宏观角度来看，Transformer与Seq2Seq的结构相同，依然引入经典的Encoder-Decoder结构，只是其中的神经层已经不是以前的RNN和CNN，而是完全引入注意力机制来进行构建。

![stack](F:\video\stack.png)

上图代码的复现结构

```python
class EncoderDecoder(nn.Module):
    """
        整体来说Transformer还是Encoder和Decoder结构，其中包括两个Embedding，一块Encoder，一块Decoder，一个输出层
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

接下来，我将按照结构顺序一一介绍

## Embedding

这一层没什么好说的倒是，就是一个Embedding层。

```python
class Embeddings(nn.Module):
	"""
		将单词转化为词向量
	"""
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

```

**困惑**：为什么要乘以sqrt(d_model),希望有大神给予指点！

## Positional Emcoding

由于Transfomer完全引入注意力机制，其不像CNN和RNN会对输入单词顺序自动打上标签，其无法输出每个单词的顺序，在机器翻译中可是爆炸的啊，举个例子，你输入一句：我欠你的一千万不用还了，他返回一句：你欠我的一千万不用还了，那不是血崩。

为了解决这个问题，Transfomer在Encoder过程中为每个输入单词打上一个位置编码，然后在Decoder将位置变量信息添加，该方法不用模型训练，直接按规则进行添加。

我们看看原论文里的图

![Emcodeing](F:\video\Emcodeing.png)

其计算的具体公式在原论文3.5，

![Emcoding公式](F:\video\Emcoding公式.png)

对于上述公式，代码中进行了对数转化，具体如下

```python
class PositionalEncoding(nn.Module):
    """
    	位置变量公式详见https://arxiv.org/abs/1706.03762(3.5)
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # pe 初始化为0，shape为 [n, d_model] 的矩阵，用来存放最终的PositionalEncoding的
        pe = torch.zeros(max_len, d_model)
        # position 表示位置，shape为 [max_len, 1]，从0开始到 max_len
        position = torch.arange(0., max_len).unsqueeze(1)
        # 这个是变形得到的，shape 为 [1, d_model//2]
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        # 矩阵相乘 (max_len, 1) 与 (1, d_model // 2) 相乘，最后结果 shape   (max_len, d_model // 2)
        # 即所有行，一半的列。（行为句子的长度，列为向量的维度）
        boy = position * div_term
        # 偶数列  奇数列 分别赋值
        pe[:, 0::2] = torch.sin(boy)
        pe[:, 1::2] = torch.cos(boy)
        # 为何还要在前面加一维？？？
        pe = pe.unsqueeze(0)
        # Parameter 会在反向传播时更新
        # Buffer不会，是固定的，本文就是不想让其被修改
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x能和后面的相加说明shape是一致的，也是 (1, sentence_len, d_model)
        # forward 其实就是，将原来的Embedding 再加上 Positional Embedding
        x += Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
```

## Encoder

Encoder代码结构

```python
class Encoder(nn.Module):
    """核心Encoder是由N层堆叠而成"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        将输入x和掩码mask逐层传递下去，最后再 LayerNorm 一下。
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

Encoder块主要就是利用Clone函数重复构建相同结构的Encoder_layer.

### Clone

整个Encoder部分由N个Encoder_layer堆积二乘，为了复刻每个层的结构，构建克隆函数。(Decoder类似)

```python
def clones(module, N):
    """
    	copyN层形成一个list
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

### Encoder_layer

接下来，我们在将独立的Encoder_layer进行拆分，看看里面到底是什么东西。

![encoderlayer](F:\video\encoderlayer.png)

上图很清晰的展示Encoder_layer内部的结构，X1，X2是经过转化的词向量，经过Positional Emcoding进入Encider_layer,喝口水，这部分要讲的有点多……

首先，Encoder_layer分为上下两层，第一层包含self-attention+SublayerConnection,第二层为FNN+SublayerConnection。Attention的内容我后面再说，这里先讲下什么是SublayerConnection。attention的内容我后面再说，这里先讲下什么是SublayerConnection。

**SublayerConnection**

SublayerConnection内部设计基于两个核心：

1. Residual_Connection（残差连接），这是上图的Add
2. Layer Normalize(层级归一化)，这是上图的Normalize

如下图所示，残差连接，就是在原图正常的结构下，将X进行保留，最终得出的结果是X+F(x).

**优势**：反向传播过程中，对X求导会多出一个常数1，梯度连乘，不会出现梯度消失(详细的内容等我以后探究一下)

![Residual connection](F:\video\Residual connection.png)

具体代码如下：

```python
class SublayerConnection(nn.Module):
    """
		A residual connection followed by a layer norm.Note for code simplicity the norms is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

```

**Layer Normalize**

归一化最简单的理解就是将输入转化为均值为0，方差为1的数据，而由于NLP中Sequence长短不一，LN相较于BN在RNN这种类似的结构效果会好很多。其具体的概括就是：对每一个输入样本进行归一化操作，保证该样本在多维度符合均值0，方差为1.

```python
class LayerNorm(nn.Module):
    """构建一个 layernorm层，具体细节看论文 https://arxiv.org/abs/1607.06450
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

```

**FNN**

前馈神经网络，这就不细说了，具体代码如下：

```python
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

```

放张图，清晰一点

![FNN](F:\video\FNN.png)

## Decoder

与Encoder结构类似，其由N个Decoder_layer组成(Transformer中N=6),相较于Encoder,其接受的参数多了Encoder生成的memory以及目标句子中的掩码gt_mask.

```python
class Decoder(nn.Module):
	"""
		Gener is N layer decoder with masking
	"""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

### Decoder layer

下图很好的表明了Decoder_layer的区别

![Decoder](F:\video\Decoder.png)

简单的解释一下，每一个Decoder_layer有三层组成

第一层：Self-Attention+SublayerConnection

第二层：Encoder-Decoder Attention+SublayerConnection(Decoder独有)

第三层：FFN+SublayerConnection

具体代码如下：

```python
class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

### Mask

**作用：**Mask简单来说就是掩码的意思，在我们这里的意思大概就是对某些值进行掩盖，不使其产生效果。

文中的Mask主要包括**两种**：

- src_mask(padding_mask):由于Sequence长短不一，利用Padding对其填充为0，然后在对其进行attention的过程中，这些位置没有意义的，src_mask的作用就是不将Attention机制放在这些位置上进行处理，这就是padding_mask,其基于作用于所有attention。
- tgt_mask(sequence_mask):对于机器翻译而言，采用监督学习，而原句子和目标句子对输入模型进行训练，那就需要确保，Decoder在生成单词的过程中，不能看到后面的单词，这就是sequence_mask，其主要在Decoder中的self-attention中起作用。

具体代码：

```python
def subsequent_mask(size):
    """
    	Mask out subequent position
    	这个是tgt_mask,他不让decoder过程中看到后面的单词，训练模型的过程中掩盖翻译后面的单词。
    """
    print(subsequent_mask(3))
    tensor([[[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]]], dtype=torch.uint8)

    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

### Attention

Transfomer的最大创新就是两种Attention，Self-Attention，Multi-Head Attention

Self-Attention

作用：简单来说，就是计算句子里每个单词受其他单词的影响程度，其最大意义在于可以学到语义依赖关系。

计算公式：

![1588909761573](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1588909761573.png)

不好理解，上图

![softmax](F:\video\softmax.png)

再看看原论文的图

![scaled dot attention](F:\video\scaled dot attention.png)

再来看看代码：

```python
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)
```

基本参照看公式应该可以很好的看懂这些代码，具体的步骤各位就参考原论文吧。

**Multi-Head Attention**

多个Self-Attention并行计算就叫多头计算（Multi-Head Attention）模式，也就是我们俗称的多头怪，你可以将其理解为集成学习，这也是Transformer训练速度超快的原因。

废话不说，上图：

![Multi-head](F:\video\Multi-head.png)

首先，我们接进来单词转化为词向量X(n,512)和W^Q, W^k, W^v(512,64)相乘，得出Q,K,V(n,64)，就生产一个Z(n,64),然后利用8个头并行操作，就生产8个Z(n,64).

![multi-head1](F:\video\multi-head1.png)

然后将8个Z拼接起来，Z就变成了(n,512)内积W^0（512，512），输出最终的Z，传递到下一个FNN层。

![Multi-Head 结构层](F:\video\Multi-Head 结构层.png)

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

```

## Generator

就是数据经过Encoder-Decoder结构后还需要一个线性连接层和softmax层，这一部分预测层，命名为Generator。

```python
class Generator(nn.Module):
    """
    	定义输出结构，一个线性连接层+softmax层
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

```

好了，这基本上把整个Transfomer进行了一个非常细致的拆分。Transformer可以说现在所有最新模型的基础，对于这一部分还是需要好好理解。

# 参考链接

- [1] [原论文](https://arxiv.org/pdf/1706.03762.pdf)
- [2] [NLP国外大牛(Jay Alammar)详解Transformer](https://jalammar.github.io/illustrated-transformer/)
- [3] [哈佛NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html#attention)
- [4] [Blog](https://juejin.im/post/5b9f1af0e51d450e425eb32d#heading-10)
- [5] [Csdn](https://blog.csdn.net/baidu_20163013/article/details/97389827)



































