## Usage

- 环境
    - pytorch 0.3
    - python 2.7x

- （已经处理过）数据预处理: `python dataset/semeval.py`去生成npy文件
- 训练/测试
    ```
    python main_sem.py train
    ```
- 结果位于`semeval`文件夹，使用`./test.sh` 进行semeval的测试，最后结果保存于res.txt
