---
title: GNN系列之_GraphSAGE
categories:
  - 深度学习
tags:
  - GNNs
mathjax: true
date: 2021-08-17 21:34:49
description:
---
# GraphSAGE

## 定义

17年**Hamilton**在GCN发布不久以后就发布了这篇文章，原文是《Inductive Representation Learning on Large Graphs》，从题目很明显可以看出，该论文强调两个事情：1.inductive；2.Large Graphs。

1.什么是inductive

在常用的机器学习或者深度学习模型中，我们通常会讲数据集分为训练集，测试集，验证集，各个集合之间相互是独立的，因为如果存在交集就变成了数据泄漏，那测试集的效果就不能正确的反应结果。但是在GCN中，由于模型中存在领结矩阵，这个是数据集通用的，这样的训练方式叫做transductive。因为为了避免这种类似数据泄漏的操作，GraphSAGE是一种inductive模式，即训练集，测试集，验证集相互独立。

2.为什么叫适用于大规模图

从GCN的公式中我们可以知道，GCN的训练需要将全部的领结矩阵放入训练，这样对于大规模图训练是不可用的，而GraphSAGE是利用采样聚合的方式，训练聚合函数，因此可以用minbatch来训练大规模图。

3.为什么叫GraphSAGE

这个是我在一开始就想问的，一个图表示训练模型为什么取这个名字，后来看论文才知道，SAGE取自两个单词：(SAmple and aggreGatE），也是简单的表明该模型的两个特色。

## 实现步骤

**伪代码**

![伪代码](/images/GraphSAGE.png)

参数解释：

- K：层数
- AGGREGATE：聚合函数，有3种
- concat：拼接矩阵

个人理解：输入初始特征矩阵（可以是one-hot/随机初始化），经过K层聚合矩阵，其实也是聚合了K步的领结信息，利用某种**聚合**函数，将每个节点的特征和其**采样**的领结节点特征进行融合。

**损失函数**
$$
J_{g}(z_{u})=-log(\sigma (z_{u}^{T}z_{v}))-Q\cdot E_{v_{n}\sim P{_{n}}^{(v)}}log(\sigma (-z_{u}^{T}z_{v_{n}}))
$$

- $z_{u}$为节点u通过GraphSAGE生成的embedding。
- 节点v是节点u随机游走访达“邻居”。
- $v_{n}\sim P{_{n}}$表示负采样：节点$v_{n}$是从节点u的负采样分布 ![[公式]](https://www.zhihu.com/equation?tex=P_n) 采样的，Q为采样样本数。

简单理解就是希望节点u与“邻居”v的embedding也相似（对应公式第一项），而与“没有交集”的节点 ![[公式]](https://www.zhihu.com/equation?tex=v_n) 不相似（对应公式第二项)。

## 聚合函数

### Mean aggregator

**平均聚合**
$$
\begin{matrix}
h_{N(v)}^{k}=mean(\{h_{u}^{k-1},u\in N(v)\})
\\ 
h_{v}^{k}=\sigma (W^{k}\cdot CONCAT(h_{v}^{k-1},h_{N_{(u)}}^{k}))
\end{matrix}
$$
就是伪代码写的那种，先对k-1采样的领结节点特征进行求平均，然后和K-1层的节点进行拼接，在利用参数Wk进行纬度转换。

**归纳式聚合**
$$
h_{v}^{k}=\sigma (W^{k}\cdot mean(\{h_{v}^{k-1}\}\cup \{h_{u}^{k-1},\forall u\in N(v) \}))
$$
直接对k-1层，v节点+采样的领结节点特征进行求平均，利用参数$W^{k}$进行纬度转换。

### LSTM

对领结节点进行随机排序，因为采样的LSTM是固定的，然后作为序列放入LSTM最后输出一个embedding就是v。

### Pooling

$$
Aggregate_{k}^{pool}=max(\{\sigma (W_{pool}h_{u_{i}}^{k}+b),\forall u_{i}\in N(v)\})
$$

把各个邻居节点单独经过一个MLP得到一个向量，最后把所有邻居的向量做一个max-pooling或者mean-pooling来获取。

## 总结

**优点：**

1. GraphSAGE基于采样+聚合的策略，可以很好的解决GCN将整个邻接矩阵放入训练导致内存溢出的问题，可以用于大规模图中。
2. GCN不能去推测没有看到的节点，因为他的训练依赖邻接矩阵，而GraphSAGE训练的是一个聚合函数，所以他可以用已只节点去推测未知节点，前提是未知节点的领结节点存在于GraphSAGE中。

**不足**：

1. 他既然是聚合函数，没有用到Attention，也就是说对于权重的分配没有采取更好的策略。因此才诞生了GAT。







