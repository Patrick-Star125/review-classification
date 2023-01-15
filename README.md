# 基于LSTM的外卖评论分类

## 基本介绍

### 任务简介

本项目基于公开的模型仓库 MindSpore/LSTM 进行二次开发，通过外卖评论数据集进行分类任务。

任务链接：https://xihe.mindspore.cn/competition/text_classification/0/introduction

#### LSTM简介
在自然语言处理中常用RNN网络，但RNN细胞结构简单，容易在训练中产生梯度消失问题。例如RNN网络在序列较长时，在序列尾部已经基本丢失了序列首部的信息。为了克服这一问题，LSTM(Long short term memory)模型被提出，通过门控机制来控制信息流在每个循环步中的留存和丢弃。下图为LSTM的细胞结构拆解：

<img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/lstm/lstm-model.PNG" width="80%">

#### 数据集简介
数据集为外卖评论数据集，包含约4000条正向评论、8000条负向评论。 正向评价为1，负向评价为0。

| Review                                       | Label |
| -------------------------------------------- | ----- |
| 隔三差五儿的就点两杯味道不错！送餐员辛苦啦！ | 1     |
| 牛肉非常非常老，腥味太重                     | 0     |

测试集包含1000条无标注评论，对其进行推理并提交评测

### 项目结构

项目的目录分为两个部分：推理（inference）和训练（train），推理相关的代码放在inference文件夹下，训练相关的代码放在train文件夹下。

```python
 ├── inference    # 推理可视化相关代码目录
 │  ├── app.py    # 推理文件，运行产生测试集结果，可直接提交评测
 │  ├── pip-requirements.txt    # 推理可视化相关依赖文件
 │  ├── config.json    # 推理相关模型文件路径配置
 │  └── result    # 测试集推理结果
 │
 └── train    # 在线训练相关代码目录
    ├── pip-requirements.txt  # 训练代码所需要的package依赖声明文件
    ├── lstm_aim_cust.py  # 自定义Aim训练代码 
    ├── train.py       # 训练代码
    ├── save_model     # 每一轮迭代测试集推理结果
    └── result        # 每一轮迭代测试集推理结果
```

### 使用方法

预训练词向量为微博词+字300d，下载仓库：https://github.com/Embedding/Chinese-Word-Vectors

训练过程

~~~bash
python train.py --rw_path mindcon_text_classification --weibo_path dataset\sgns.weibo.char --epochs 0,40
~~~

推理过程

~~~bash
python app.py --rw_path mindcon_text_classification --weibo_path dataset\sgns.weibo.char --model_path save_model\sentiment-analysis.ckpt
~~~

在昇思大模型平台有同样的项目，可以直接训练

## 效果展示

### 训练

ModelArts平台

![](http://pic.netpunk.space/images/2023/01/09/20230109184456.png)



### 评估

![img](https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/resnet50/aim_metrics.png)






