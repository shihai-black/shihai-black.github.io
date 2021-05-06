---
title: 多进程、多线程
date: 2021-04-11 11:23:23
categories:
- python
tags:
description:
- 此文档写于2020年，建成于博客创立之前。
---

> ​	这周接到1个需求，项目经理觉得我这边构造数据太过缓慢，由于数据量过大，以前数据构造完后将其导入MPP，利用copy_from速度还是很快的，一般为10W/s。现在换成Kafka消息队列，又由于python库自带的原因（这个是组里大神告诉我的），无法像java开发利用list导数据，因此我只能一条一条以json的形式将数据放入消息队列，一般为800条/s。原先构造5000W数据需要7H，经过优化后暂时需要3.5H，性能提升一倍。有点小开心，话不多说，石海同学开始将一下个人对于标题的理解。。。。
>
> ​	进一步优化利用多进程的JoinableQueue，一边一直产生数据，另一边一直消费数据，现在5000W数据大约需要1.5H，性能较最初提升了近五倍。

## 历史

上古年代：在很久很久以前，当时主流的磁盘操作系统MS-DOS是只支持单任务操作的，就打个比方，如果我想在电脑上听音乐和看电影，是不能同时开启的，只能先听音乐后看电影，或者位置互换。

2002：横空出世的Intel Pentium 4HT处理器，提出了cpu多线程（SMT），其支持一个cpu进行多任务开启。

————你总不能要求Guido1989年为了打发圣诞节假期发明的一种编译语言还要设计一下多线程的部分。

2006：在秋季的英特尔信息技术峰会上，Inter总裁宣布，其在11月份将会交付第一台4核cpu处理器，而这距离双核发布还不到1年，支持多核时代就此拉开序幕。

————为什么说python多线程是历史遗留问题，因为在当时想要在两个或者更多的处理器对同一个对象运行时，为了保护线程数据完整性和状态同步，最简单的方法就是加锁，于是出现了GIL这把超级大锁。

2006-至今：GPU也浩浩荡荡发展了十几年了，过程我就不细说了（因为不懂），总之对于图象类任务GPU和CPU不是一个量级上的，以前看过一个视频，很好的解释了CPU和GPU的区别。比如我要画一幅图，cpu需要一笔一划将他画出来，而GPU是直接在脑子里构思好，一炮就把图打出来了。因此对于神经网络这个模型，多个神经元同时进行计算，GPU比CPU快太多。。。。。

## 编译器、解释器、IDE

**编译器**：对于C、C++这类语言需要先编译后运行，其中编译的时候就是编译器。

**解释器**：对于python、PHP、JavaScript等就是利用解释器，’一边编译一边解释‘，不过速度会比较慢，因此产生一种叫预编译的东西，python在运行时就先生成pyc的二进制临时文件，在出结果。

预编译：Java、C#，运行前将源代码编译成中间代码，然后再解释器运行，这种执行效率比编译运行会有效率一些，避免重复编译。

**IDE**：集成[开发环境](https://baike.baidu.com/item/开发环境)（[IDE](https://baike.baidu.com/item/IDE)，Integrated Development Environment ），用于提供程序开发环境的应用程序，python常用的就是pycharm和jupyter notebook。

## GIL锁

**概念：**GIL全称Global Interpreter Lock,其并不是python的特性，由于Cpython是大部分环境下默认的python执行环境，而在实现python解释器会自动上锁。

**目的：**确保每个进程只有一个线程运行，所以在外面我们一般说python的多线程是伪线程。因为不管你有几个核，你用多线程也只能跑一个核。

## 进程、线程、协程的利用

**进程**：拥有代码和打开的文件资源、数据资源、独立的内存空间。

**线程**：从属于进程，是程序的实际执行者，线程拥有自己的栈空间。

**协程**：从属于线程，是一种比线程更加轻量级的存在。

**总结：**

**对操作系统来说，线程是最小的执行单元，进程是最小的资源管理单元。**

## 多进程

我觉得最简单的讲，多进程就是你在任务窗口看建了几个任务。比如你电脑上有16个核，理论上你可以开16个进程，每个进程占满一个cpu。对python而言由于没有多线程的利用，所有在单进程无法满足需求时，自然得利用多进程。

### 单任务单进程

```python
from multiprocessing import Process


def work(name):
    print(f'{name}')


if __name__ == '__main__':
    p = Process(target=work,args=('single task',))
    p.start()
    p.join()

```

我还看到一种是用run不用start的，但大概看了下没什么特殊的，不过这个join只能用于start.所以建议还是都用start，这个join是阻塞的意思，简单来说：主进程和其他子进程结束的话，都等着，等我结束了才能结束。

举个例子：

```python
from multiprocessing import Process
import os
import time

def work1(name):
    for i in range(5):
        print(f'{name}:{os.getpid()}')
        time.sleep(2)


def work2(name):
    for i in range(5):
        print(f'{name}:{os.getpid()}')


if __name__ == '__main__':
    p1 = Process(target=work1,args=('single task1',))
    p2 = Process(target=work2,args=('single task2',))
    print(f'主进程:{os.getpid()}')
    p1.start()
    p1.join()
    p2.start()
    p2.join()
```

如果这么写，主进程你走你的，work2子进程你给我等一下。等我run完你在run。

**注意：**每个子进程之间的join不是串行的，是并行的。既无论你多少个子进程，谁最后结束，谁关闭文件。

### 单任务多进程

比如你要启动50个进程，然后写50个子进程就太慢了。因此，对于多进程有一种新的形式。

```python
def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


########################################################
#运行结果
########################################################
Parent process 7748.
Waiting for all subprocesses done...
Run task 0 (7780)...
Run task 1 (7781)...
Run task 2 (7782)...
Run task 3 (7783)...
Run task 4 (7781)...
All subprocesses done.
```

可以看到task1和task4的进程是一样的，因为只启动了4个进程，因此第5个进程需要等待其他进程结束才开始运行。

**注意：**close不加直接写join是会报错的。

### 多任务多进程

我这边用的是经典的生产者和消费者模型，及一个模型构造生产者，一个模型构造消费者，两者通过自带的JoinableQueue进行通信。（不要用deque里面的队列！！）

```python
import time
import random
from  multiprocessing import JoinableQueue,Process
def producer(q,name,food):
    for i in range(5):
        time.sleep(random.random())
        fd = '%s%s'%(food,i+1)
        q.put(fd)
        print('%s生产了一个%s'%(name,food))
    q.join()  # 我启动了生产者之后，生产者函数一直在生成数据，直到生产完所有数据将队列q.join()一下，意思是当我生产的数据都被消费者消费完之后 队列的阻塞才结束。
def consumer(q,name):  # 消费者不需要像Queue那样判断从队列里拿到None再退出执行消费者函数了
    while True:
        food = q.get()
        time.sleep(random.random())
        print('%s吃了%s'%(name,food))
        q.task_done()  # 消费者每次从队列里面q.get()一个数据，处理过后就使用队列.task_done()
if __name__ == '__main__':
    jq = JoinableQueue()
    p =Process(target=producer,args=(jq,'喜洋洋','包子'))
    p.start()
    c = Process(target=consumer,args=(jq,'灰太狼'))
    c.daemon = True  # 守护进程，如果用Pool就不用这个也没事
    c.start()
    p.join()  # 阻塞生产者
```

具体实现情况

1. 启动一个生产者，和一个消费者（这里可以用多进程消费），看具体时间
2. 生产者结束后将JoinableQueue进行阻塞，直到队列全部被消费后，才结束进程。

Queue与JoinableQueue的区别

1. Queue在消费者中需要if判断队列，不然的话就陷入死循环。而jionableQueue不用，其只需要将队列堵塞后，队列消费完程序才解决。
2. Queue很难控制，因为如果只是判断队列是否存在，就需要考虑产生和消耗队列的时间。因此我认为还是JoinableQueue好

## 多线程

开头就说了，python虽然线程是真正的线程，但在解释器执行的时候会遇到GIL锁，导致其无论多少核，只能跑满一个核，所以对于比较简单的可以运行多线程，好像在爬虫里面运用多线程的比较多。

那为什么有的程序一个进程一开用了N个cpu，那是因为他进程中存在利用c扩写的库，他将关键的部分用C/C++写成python扩展，其他还用python写。一般计算密集型的程序都会用C代码编写并通过扩展的方式集成到python脚本（numpy）。在扩展中完全可以用C创建原生线程，而且不用锁GIL，充分利用CPU的计算资源。

```python
from threading  import Thread
import time

def work1():
    print('work1 has working')
    time.sleep(3)

def work2():
    print('work2 has working')
    time.sleep(5)

def work3():
    print('work3 has working')
    time.sleep(8)

if __name__ == '__main__':
    start_time =time.time()
    p1 = Thread(target=work1)
    p2 = Thread(target=work2)
    p3 = Thread(target=work3)
    p1.start()
    p2.start()
    p3.start()
    p3.join()
    print(f'cost time {time.time()-start_time}')
```

## 问题

```
能否利用Pool批量构造生产者和消费者，
暂时只能做到Pool构造消费者，多个Process构造生产者进行生产。。。
（除非使用process分别构造生产者和消费者模型）
```









