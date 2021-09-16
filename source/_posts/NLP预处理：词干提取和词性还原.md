---
title: NLP预处理：词干提取和词性还原
categories:
  - 特征工程
tags:
  - NLP
date: 2021-06-04 14:45:54
description:
---

> 最近在做的项目里面需要做一个关键词多属性提取，然后由于以前接触的都是中文的项目，所以就会拿中文分词的一些操作来用，但在英文的项目，其实就有词干提取和词性还原两种非常好用的方法

自然语言处理中一个很重要的操作就是所谓的stemming 和 lemmatization，二者非常类似。它们是词形规范化的两类重要方式，都能够达到有效归并词形的目的，二者既有联系也有区别。

## 词干提取

NLTK中提供了三种最常用的词干提取器接口，即 Porter stemmer, Lancaster Stemmer 和 Snowball Stemmer。而且词干提取主要是基于规则的算法。

**Porter**

1980年最原始的词干提取法

```python
from nltk.stem.porter import PorterStemmer
text='chineses'
porter_stemmer = PorterStemmer()
stem_text = porter_stemmer.stem(text)
print(stem_text)
--------------------
Out[148]: 'chines'
```

**Snowball**

也称为Porter2.0，本人说提取的比1好，这个不清楚，但确实相比较1.0会快一些，感觉上效率有提升1/3.

```python
from nltk.stem.snowball import EnglishStemmer
text='chineses'
porter_stemmer = PorterStemmer()
stem_text = porter_stemmer.stem(text)
print(stem_text)
--------------------
Out[148]: ‘chines’
```

**Lancaster**

据说比较激进，没什么人用过

```python
from nltk.stem import LancasterStemmer
text='chineses'
lancas_stemmer = LancasterStemmer()
stem_text = lancas_stemmer.stem(text)
print(stem_text)
--------------------
Out[148]: ‘chines’
```

## 词性还原

基于字典的映射，而且在NLTK中要求标明词性，否则会出问题。一般需要先走词性标注，再走词性还原，因此整体链路相较于词干提取较长。

```python
from nltk.stem import WordNetLemmatizer  
lemmatizer = WordNetLemmatizer()  
word = lemmatizer.lemmatize('leaves',pos='n') 
print(word)
--------------------
Out[148]: 'leaf'
```

