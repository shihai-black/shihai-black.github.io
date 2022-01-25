---
title: 对比学习_SimCSE
categories:
  - 深度学习
tags:
  - 对比学习
mathjax: true
date: 2021-09-30 17:10:39
description:
---
# 对比学习之SimCSE

> 前短时间接到一个title i2i召回数据的任务，最开始只是想用简单的Doc2vec输出每个title的emb然后用faiss做召回，但是效果很差。后来尝试了用Bert直接输出结果，效果也很差。后面看论文发现Bert的emb存在模型坍塌问题【详细可以参考美团的ConSERT任务】。最后利用SimCSE比较合理的解决了需求

## 问题

1. 什么是自监督学习，什么是对比学习？
2. SimCSE生成监督数据的方法是什么？
3. 为什么生成的句向量这么有用？
4. 他的优劣势是什么？

## 概念

### 自监督学习

首先，我们知道现在训练模型的方式主要是两种：有监督和无监督，并且我个人认为无监督的使用场景是远远大于有监督的。无监督有一个比较特别的分支是自监督，他的特点就是不需要人工标注的类别标签信息，直接利用数据本身作为监督信息，来学习样本数据的特征表达。

自监督主要分为两类：

- Generative Methods：其中的典型代表就是自编码器，其利用数据本身压缩再重构去比较生成的数据与原始数据的差距。
- Contrastive Methods：这类方法则是通过将数据分别与正例样本和负例样本在特征空间进行对比，来学习样本的特征表示。

**备注**：我个人任务word2vec也可以属于自监督范畴，不过不属于上面两类，他是利用文本词与词之间关系去学习每个单词的embedding

### 对比学习

上面说了，对比学习属于自监督学习的一个大类，他主要的难点就是如何生成相似的正样本。其最开始是在cv中使用的，后面NLPer也不甘示弱的进行研究，才有了今天NLP在对比学习领域的百花齐放，具体的进展可以看[张俊林博客](https://zhuanlan.zhihu.com/p/367290573)。

SimCSE全称(Simple Contrastive Learning of Sentence Embeddings)，是一种在没有监督训练数据的情况下训练句子向量的对比学习框架。并且从无监督和有监督两种方法给出了结果。

- 无监督SimCSE：仅使用dropout进行数据增强操作。具体来说，将同一个样本输入预训练编码器两次(BERT)，由于每次的dropout是不同的，那么就会生成两个经过dropout后的句向量表示。这里的dropout主要是指传统的feature dropout，在标准的Transformer实现中，每个sub-layer后都默认配置了dropout。除此之外，Transformer也在multi-head attention和feed-forward network的激活函数层添加了dropout。利用dropout将这两个样本作为“正样本对”，于此同时，同一个batch里面不同样本做"负样本对"
- 有监督SimCSE：基于natural language inference (NLI) 数据集进行训练，具体来说，论文将entailment样本作为“正样本对”，并将contradiction样本作为hard“负样本对”，并且从实验证明该方法是有效的。

## 指标详解

在《**Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**》一文中，作者指出了现有的对比学习都遵循以下两大原则即：相似样本应该保持靠近，并且不同的样本应该分散嵌入在高维球体。于此同时SimCSE的作者也提到了这两点，他也认为这两点是衡量表示质量很重要的指标。

### alignment

用于衡量两个样本的相似程度，其计算公式如下：
$$
L=E\left \| f(x)-f(y)\right \|_{2 }^{\alpha}
$$
同时作者给出了在pytorch中的代码

```python
def lalign(x, y, alpha=2):
  return (x - y).norm(dim=1).pow(alpha).mean()
```

### uniformity

衡量规整后的特征在unit 超球体上的分布的均匀性，其计算公式如下：
$$
L=log(E[e^{-t\left \| f(x)-f(y)\right \|^{2}}])
$$
同时作者给出了在pytorch中的代码

```python
def lunif(x, t=2):
  sq_pdist = torch.pdist(x, p=2).pow(2)
  return sq_pdist.mul(-t).exp().mean().log()
```

两者结合，作者是介绍可以直接作为损失进行优化的：

```python
loss = lalign(x, y) + lam * (lunif(x) + lunif(y)) / 2
```

## 损失函数

### unsup

$$
L=-log\frac{e^{sim(h_{i},{h_{i}}^{+})/\tau }}{\sum_{j=1}^{N}e^{sim(h_{i},{h_{j}}^{+})/\tau}}
$$

-  $h_{i}$表示查询向量，+表示对比向量，j表示大小为N的batch下其他向量
-  $\tau$表示温度参数，默认是0.07，[详解温度参数](https://zhuanlan.zhihu.com/p/357071960)
- sim()表示余弦相似度，当然用内积也是一样的，因为l2标准化后的内积等价于余弦相似度

### Sup

$$
L = -log\frac{e^{sim(h_{i},{h_{i}}^{+})/\tau}}{\sum_{j=1}^{N}(e^{sim(h_{i},{h_{j}}^{+})/\tau}+e^{sim(h_{i},{h_{j}}^{-})/\tau})}
$$

- 其他和无监督方法没有区别，只是将负向量也纳入分母

**备注**：其实他的目标就是为了让分子尽可能的小，分母尽可能的大，那分子其实就代表alignment指标，分母就是代表uniformity指标，其实简单看和交叉墒损失函数很像，并且在Sentence transfomer包中的实现也是这样子的。

## 代码详解

### SimCSE包

```python
from simcse import SimCSE
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
sentences = ['A woman is reading.', 'A man is playing a guitar.']
model.build_index(sentences)
results = model.search("He plays guitar.")
```

作者直接将训练好的模型放在huggingface上，并且下游利用Faiss进行向量搜索，详情可以直接看github代码，如果想自己训练也很简单，利用Sentence_transfomer包有现成的demo

### Sentence_transfomer包

**unsup**

```python
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from torch import nn
from datetime import datetime
import os
import gzip
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
model_name = 'distilbert-base-uncased'
train_batch_size = 128
num_epochs = 1
max_seq_length = 32
model_save_path = 'output/training_stsb_simcse-{}-{}-{}'.format(model_name, train_batch_size,
                                                                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
sts_dataset_path = 'data/stsbenchmark.tsv.gz'
if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length,
                                          cache_dir='../distilbert-base-uncased')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense = models.Dense(pooling_model.pooling_output_dimension,512,activation_function=nn.ReLU())  # 降维操作
model = SentenceTransformer(modules=[word_embedding_model, pooling_model,dense],device="cuda:1")
wikipedia_dataset_path = 'data/wiki1m_for_simcse.txt'
if not os.path.exists(wikipedia_dataset_path):
    util.http_get(
        'https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt',
        wikipedia_dataset_path)
print('read wiki data')
train_samples = []
with open(wikipedia_dataset_path, 'r', encoding='utf8') as fIn:
    for line in fIn.readlines()[:10000]:
        line = line.strip()
        if len(line) >= 10:
            train_samples.append(InputExample(texts=[line, line]))
print(len(train_samples))
print('Read STSB dev dataset')
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                 name='sts-dev')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
                                                                  name='sts-test')

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1) #Evaluate every 10% of the data

dev_evaluator(model)

print('Start train model')

# train the model
model.fit(train_objectives=[(train_dataloader,train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr':5e-5},
          use_amp=False
          )

model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)

```

**sup**

```python
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv

# argument
model_name = 'distilbert-base-uncased'  # 可以自行替换
train_batch_size = 32
num_epochs = 1
max_seq_length = 64

# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=max_seq_length,
                                          cache_dir='../distilbert-base-uncased')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Use label sentences from NLI dataset train to train out model

nli_dataset_path = 'data/AllNLI.tsv.gz'

if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

print('Read NLI dataset')
model_save_path = 'output/simcse-{}-{}-{}'.format(model_name, train_batch_size,
                                                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
train_samples = []
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    count = 0
    for row in reader:
        count += 1
        label = row['label']
        if label == 'contradiction':
            premise = row['sentence1']
            negative = row['sentence2']
        elif label == 'entailment':
            premise = row['sentence1']
            positive = row['sentence2']
        if count % 3 == 0:
            if row['split'] == 'train':
                train_samples.append(InputExample(texts=[premise, positive, negative]))
print(f'train sample length:{len(train_samples)}')

# Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                 name='nli-dev')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
                                                                  name='nli-test')
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1)  # Evaluate every 10% of the data

print('Start training')
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr': 5e-5},
          use_amp=True  # Set to True, if your GPU supports FP16 cores
          )

print('Start test')
model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)
```

sup在demo中没有，只是训练数据的构造需要自己去改写一下。如果是简单的复现，用上面几个包就可以了，避免重复造轮子，但如果想对组件进行相应的替换，那就得自己写，我以后有时间也想尝试复现一下。

## 实际对比效果

**query**:Sexy Elegant Base Tanks & Camis Female Smooth Sling Dress

**result**：

| Doc2vec                                                      | SimCSE                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Glossy Stainless Steel Simple Laser Open Bracelet            | Sexy Tanks & Camis Female Viscose Sling Dress                |
| Elegant Small Smile Face Cute Letter Earrings                | Waisted Sexy Fitting Base Tanks & Camis Sling Dress          |
| Elegant Simple Cross Shell Beads Earrings                    | Sexy Base Tanks & Camis Summer Female Sling Dress            |
| Sexy Takato Ultra-thin Lace Lace Sexy Stockings              | Waisted Split Elegant Base Tanks & Camis Female New Style Sling Dress |
| Magnet Pillow Core Will Gift Magnet Gift Pillow              | Waisted Base Tanks & Camis Vintage Female Sling Dress        |
| Fitting Suit Collar Short Sleeve Spring Summer High Waist Female Knit Dress | Elegant V-neck Satin Fitting Thin Sling Dress Female Tanks & Camis External Wear Dress |
| Elegant Heart Simple Small Piercing Jewelry                  | Sexy Off-the-shoulder Casual Pirinted Sling Dress            |
| Sexy Dress Women's One Shoulder Slit Long Skirt              | Thin Long Split Sexy Fitting Elegant Base Tanks & Camis Female Black Sling Dress |
| Leather Spring Summer Leather New Arrived First Layer Vintage Manual Simple Single Shoes | Mid-length Sexy Tanks & Camis Ultrashort Female Smooth Dress |

我只能说，对比学习在语句表达上确实和docvec不是一个量级的，太强了。。。

## 总结

1. SimCSE在语句表达上很不错，而且他好像还比美团后出的Consert还要强一些
2. 至少这篇论文说明了在模型中做数据增强，比从源头做替换/裁剪等方式数据增强效果更好
3. SimCSE有两个问题，在原作者近期的论文中【ESimCSE】提出来了，并且提出了解决方案：
   1. 由于正样本长度都是一样的，因此从长度这个特征来说，就可以区分很多样本，为了避免这个问题。原作者利用一种相对安全的方式：word repetition，就是对随机对句子进行单词重复的操作，一方面改变了正样本长度相等的缺陷，另一方面保持两者特征相似。
   2. 他承担不了大的batch-size,一方面是内存不允许，一方面是效果也会下降，因此作者仿造在cv中的Momentum Contrast的操作用于提升性能。【为了缩小train和predict的差距，ESimCSE关闭了MC encoder中的dropout】
4. 这篇论文的模型就是用的Sentence Bert，只是损失函数换了，原论文是CosineSimilarityLoss，而这篇论文改成了InfoNCE，利用Bert的dropout直接强有力的提升对比学习的效果。

## 附录

[Alignment & Uniformity](https://arxiv.org/pdf/2005.10242.pdf)

[github_simcse](https://github.com/princeton-nlp/SimCSE)

[github_sen_tran](https://github.com/UKPLab/sentence-transformers/)

[paper_simcse](https://arxiv.org/abs/2104.08821)

[paper_consert](https://arxiv.org/abs/2105.11741)

[paper_esimcse](https://arxiv.org/pdf/2109.04380.pdf)

[paper Moco](https://arxiv.org/abs/1911.05722)