---
title: Louvain算法
categories:
  - 机器学习
tags:
  - community detective
mathjax: true
date: 2021-09-16 21:49:43
description:
---

> 利用商品的点击和购买，对商品进行聚类分析，因此选择了Louvain这个高效的聚类算法，再此主要是为了记录一下，方便日后回顾

问题

- 什么是模块度，其代表什么含义，公式推导
- Louvain算法的精髓是什么，以前是怎么做社团发现的
- 社团发现和广义聚类的区别
- 为什么Louvain算法会存在resolution问题，后面是怎么解决的。

## 模块度

Newman[1](https://arxiv.org/pdf/cond-mat/0308217.pdf)，2003年首次提出了第一版模块度，后在[2](https://www.pnas.org/content/pnas/103/23/8577.full.pdf)，2006年提出了第二版模块度，经过多次修正后形成现在我们认知中的模块度。

### 解释

模块度有可以从两个角度解释，一种是较为宏观的表示簇内聚集度和簇外离散度的综合指标，另一种从数学角度认为是在给定组内的边的比例减去边是随机分布的期望分数，其具体的值属于【-0.5，1】。论文中认为0.3-0.8是较好模块度

### 定义

**第一版**

假设网络被划分为 $k$ 个社区，那么定义一个$k×k$的对称矩阵$e$，它的元素 $e_{ij}$表示社区 $i$ 和社区 $j$ 之间的边的数量。矩阵的迹$Tre=\sum e_{ii}$，也就表示了在相同社区内节点之间的边集合。显然，社区划分的好，也就是社区内部节点之间联系密集，那么该值就越高，这与我们通常对社区的认识是一致的。

但是！如果不划分社区，直接将数据作为一个团，那他的Tre就是1，这是不合理的，因此又定义了一个$a_{i}=\sum e_{ij}$，表示所有连接到社区ii的边数量。最后形成第一版的模块度函数
$$
Q=\sum (e_{ii}-ai^{2})=Tre-\left \| e^{2}\right \|
$$
**第二版**

为什么Newman要对模块度重新定义呢，因为第一版没有考虑节点度的概念，节点的度在一定程度上能够表示该节点被连接的概率，并且第二版矩阵形式可以应用在spectral optimization algorithms，具体参考[wiki](https://en.wikipedia.org/wiki/Modularity_(networks)#Matrix_formulation)。
$$
Q=\frac{\sum (A_{ij}-P_{ij})\delta_{ij}}{2m}
$$

- 2m:所有节点的度数之和，为了计算的模块度不受m的规模影响
- $A_{ij}/\delta_{ij}$:节点的领结矩阵【不考虑有权那就是1,0】
- $P_{ij}$:任意两个节点i和j连接的概率

我们将$K_i$和$K_j$表示节点i和j的度，那么
$$
P_{ij}=\frac{K_i*K_j}{2m}=K_i*\frac{K_j}{2m}
$$
$K_j/2m$表示节点j被连接的概率，因此$P_{ij}$就表示节点i和j连接的概率。并且第一版和第二版本质上互通的，两者可以直接推导成一个公式。

## 算法步骤

讲完了模块度的概念，那我们知道了模块度是用于衡量一个社团结构好坏的指标，而Louvain算法就是基于该指标，利用迭代不断优化模块度，并且其简单高效。

**具体步骤**

1. 将图中的每个节点看成一个独立的社区，因此社区的数目与节点个数相同
2. 对于每个节点，尝试将该节点分配到其相邻节点所在的社区，观察其$\bigtriangledown Q$，并记录其$\bigtriangledown Q$最大相邻节点的社区，如果$\bigtriangledown Q>0$，将该节点融入该社区
3. 重复第二步直至所有节点所在的社团模块度不在变化
4. 将所有社区压缩至一点节点，社区内节点之间的边的权重转化为新节点的环的权重，社区间的边权重转化为新节点间的边权重。
5. 重复迭代直至收敛

那为什么说Louvain算法收敛速度很快呢，是因为他是根据相邻节点进行计算的，不是从全局来进行计算的，并且越上层的时候收敛越快，并且可以按层获取对应的社团。

## 算法不足和改进

### 不足

以模块度为目标函数的优化算法会存在一个分辨率限制的问题，即：无法发现社团数量小于$(N/2)^{1/2}$的社团，这对于一些小社团是不公平的。

### 改进

主要是增加分辨率的调整，具体可以参考[3](https://arxiv.org/abs/0812.1770)，这也是在python-louvain这个包中的resolution参数的来源

## 具体代码

```python
import networkx as nx
import community
import pandas as pd
import os

class FastLouvain:
    def __init__(self, pair_path, resolution, logger=None):
        self.pair_path = pair_path
        self.resolution = resolution
        self.logger = logger

    def generate_graph(self):
        G = nx.read_edgelist(self.pair_path, create_using=nx.Graph(), nodetype=str, data=[('weight', int)])
        self.logger.info('node size :{}'.format(len(G)))
        self.G = G
        return G

    def best_community(self):
        self.logger.info('Start louvain training ……')
        partition = community.best_partition(self.G, resolution=self.resolution)
        cluster_label = set([x for x in partition.values()])
        self.logger.info(f'The number of cluster_label is {len(cluster_label)}')
        self.logger.info('Start calculate modularity_q')
        modularity_Q = community.modularity(partition, self.G)
        self.logger.info(f'modularity_Q {modularity_Q}')
        return partition

    def run(self):
        G = self.generate_graph()
        partition = self.best_community()
        return G,partition


if __name__ == '__main__':
    pair_path = '../input/click_list/0908/normal_pair_click_seq_7.csv'
    resolution = 0.5
    fast_louvain = FastLouvain(pair_path,resolution)
    G,partition = fast_louvain.run()

```

## 参考

- [博客1](https://qinystat.gitee.io/2020/01/22/Modularity/#1-1%E5%8E%9F%E5%A7%8B%E5%AE%9A%E4%B9%89-Q1)
- [博客2](https://greatpowerlaw.wordpress.com/2013/02/24/community-detection-modularity/)
- [博客3](https://blog.csdn.net/wangyibo0201/article/details/52048248)
