---
title: Graph_embedding之deepwalk
categories:
  - 深度学习
tags:
  - Graph Embedding
date: 2021-06-04 14:20:54
description:
---

### 历史

deepwalk的作者Bryan参照word2vec的思想，将文本应用到图的结构中，形成了这篇文章，内容也不复杂，也算是为后世的Graph embedding开了一个头。

### 流程图

![deepwalk](/images/deepwalk.png)

### 论文理解

**适用领域**

1. network分类
2. 异常检测

**主要内容**

1. 前提基于领域假设
2. deepwalk用的是没有权重的图
3. 作者证明了单词共现和节点共现有类似的现象，因此文本的那一套也可以用于图。
4. 当数据稀疏时或者使用低于60%数据量的数据集时效果比传统模型好

**具体步骤**

1. 首先基于节点信息和边的信息，生成底层图谱
2. 以每个node作为定长序列的起始点，利用随机游走生成定长序列，所以该随机游走也称为截断式随机游走。
3. 将输出的序列作为word2vec的输入，生成nodes embedding
4. 将类别信息和nodes embedding输入模型（LR，SVM，深度都可以）做一个多分类。

**看法**

本身word2vec和deepwalk这种生成词向量的算法都是无监督的，但是后面加上一些有监督的算法就可以合理对embedding的效果做出判断，近期看的论文总会提到link predict的方法，这和现在的节点分类提供类似的作用，而且前途也很大，利用某些半监督的算法对无监督算法进行评判。

### 参考

[DeepWalk](https://arxiv.org/abs/1403.6652)