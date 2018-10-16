2018.10.16更新:

- master分支: Python 3.5x & Pytorch 0.4+
- dev 分支: Python 2.7x & Pytorch 0.3

## Introduction
基于Pytorch复现PCNN (Zeng 2014)全监督关系抽取的代码. 

相关博客地址： [关系抽取论文笔记](http://shomy.top/2018/02/28/relation-extraction/)

另外基于远程监督的PCNN+ONE (Zeng 2015) 代码连接: [PCNN+ONE/ATT](https://github.com/ShomyLiu/pytorch-relation-extraction)

## 数据集
使用Semeval 2010的9类关系（考虑方向 19类）

## 使用方法

- Python环境
    - pytorch 0.3(后续会升级为0.4及以后版本)
    - python 2.7x
    - fire

- 数据预处理: `python dataset/semeval.py`去生成npy文件
- 训练, 自动保存最优模型(未设计验证集)
    ```
    python main_sem.py train
    ```
    其中 参数配置位于 `config.py`，可以直接指定修改，如: `python main_sem.py train --batch_size=32`
- 模型预测的结果位于`semeval`文件夹，使用`./test.sh`使用semeval官方的脚本测试，最后结果保存于res.txt 
- F1大概可以到80-81%左右，经过fine-tuning 大概到82-83%

## 参考
- PCNN: Relation Classification via Convolutional Deep Neural Network
