# Text Classification with CNN and RNN

目的：使用卷积神经网络以及循环神经网络进行中文文本分类

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

还可以去读dennybritz大牛的博客：[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

以及字符级CNN的论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

本文是基于TensorFlow在中文数据集上的简化实现，使用了字符级CNN和RNN对中文文本进行分类，达到了较好的效果。

文中所使用的Conv1D与论文中有些不同，详细参考官方文档：[tf.nn.conv1d](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d)

## 环境

- Python 3 
- TensorFlow 1.3以上
- numpy
- scikit-learn
- scipy
- ubuntu18.04

## 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

这个子集可以在此下载：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

数据集划分如下：

- 训练集: 5000*10
- 验证集: 500*10
- 测试集: 1000*10

从原数据集生成子集的过程请参看`helper`下的两个脚本。其中，`copy_data.sh`用于从每个分类拷贝6500个文件，`cnews_group.py`用于将多个文件整合到一个文件中。执行该文件后，得到三个数据文件：

- cnews.train.txt: 训练集(50000条)
- cnews.val.txt: 验证集(5000条)
- cnews.test.txt: 测试集(10000条)

## 预处理

`data/cnews_loader.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

经过数据预处理，数据的格式如下：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val | [5000, 600] | y_val | [5000, 10] |
| x_test | [10000, 600] | y_test | [10000, 10] |

## CNN卷积神经网络

### 配置项

CNN可配置的参数如下所示，在`cnn_model.py`中。

```python
class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    num_filters = 128        # 卷积核数目
    kernel_size = 5         # 卷积核尺寸
    vocab_size = 5000       # 词汇表达小

    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 10         # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
```

### CNN模型

具体参看`cnn_model.py`的实现。

大致结构如下：

![images/cnn_architecture](images/cnn_architecture.png)

### 训练与验证

运行 `python3 run_cnn.py train`，可以开始训练。

> 若之前进行过训练，请把tensorboard/textcnn删除，避免TensorBoard多次训练结果重叠。

```
2019-07-23 18:25:33.132745: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
Training and evaluating...
Epoch: 1
2019-07-23 18:25:33.357006: W tensorflow/core/framework/allocator.cc:107] Allocation of 78118912 exceeds 10% of system memory.
2019-07-23 18:25:33.411465: W tensorflow/core/framework/allocator.cc:107] Allocation of 78118912 exceeds 10% of system memory.
2019-07-23 18:25:33.465147: W tensorflow/core/framework/allocator.cc:107] Allocation of 78118912 exceeds 10% of system memory.
2019-07-23 18:25:33.518551: W tensorflow/core/framework/allocator.cc:107] Allocation of 78118912 exceeds 10% of system memory.
2019-07-23 18:25:33.572679: W tensorflow/core/framework/allocator.cc:107] Allocation of 78118912 exceeds 10% of system memory.
Iter:      0, Train Loss:    2.3, Train Acc:   7.81%, Val Loss:    2.3, Val Acc:  10.06%, Time: 0:00:02 *
Iter:    100, Train Loss:   0.67, Train Acc:  79.69%, Val Loss:    1.1, Val Acc:  70.44%, Time: 0:00:17 *
Iter:    200, Train Loss:   0.24, Train Acc:  92.19%, Val Loss:   0.65, Val Acc:  79.22%, Time: 0:00:32 *
Iter:    300, Train Loss:   0.19, Train Acc:  92.19%, Val Loss:   0.45, Val Acc:  87.12%, Time: 0:00:46 *
Iter:    400, Train Loss:   0.15, Train Acc:  96.88%, Val Loss:   0.38, Val Acc:  88.64%, Time: 0:01:08 *
Iter:    500, Train Loss:   0.15, Train Acc:  95.31%, Val Loss:   0.33, Val Acc:  89.92%, Time: 0:01:22 *
Iter:    600, Train Loss:   0.27, Train Acc:  87.50%, Val Loss:   0.31, Val Acc:  91.16%, Time: 0:01:36 *
Iter:    700, Train Loss:   0.31, Train Acc:  89.06%, Val Loss:    0.3, Val Acc:  91.08%, Time: 0:01:51 
Epoch: 2
Iter:    800, Train Loss:   0.13, Train Acc:  95.31%, Val Loss:   0.26, Val Acc:  91.52%, Time: 0:02:06 *
Iter:    900, Train Loss:  0.068, Train Acc:  98.44%, Val Loss:   0.23, Val Acc:  92.90%, Time: 0:02:22 *
Iter:   1000, Train Loss:  0.037, Train Acc:  98.44%, Val Loss:   0.24, Val Acc:  92.86%, Time: 0:02:38 
Iter:   1100, Train Loss:  0.032, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  94.14%, Time: 0:02:53 *
Iter:   1200, Train Loss:   0.03, Train Acc: 100.00%, Val Loss:   0.25, Val Acc:  93.00%, Time: 0:03:09 
Iter:   1300, Train Loss:   0.12, Train Acc:  93.75%, Val Loss:   0.19, Val Acc:  94.94%, Time: 0:03:24 *
Iter:   1400, Train Loss:   0.15, Train Acc:  95.31%, Val Loss:   0.23, Val Acc:  92.82%, Time: 0:03:39 
Iter:   1500, Train Loss:  0.077, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  95.16%, Time: 0:03:54 *
Epoch: 3
Iter:   1600, Train Loss:  0.025, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  93.04%, Time: 0:04:09 
Iter:   1700, Train Loss:  0.038, Train Acc:  98.44%, Val Loss:   0.22, Val Acc:  93.62%, Time: 0:04:24 
Iter:   1800, Train Loss: 0.0078, Train Acc: 100.00%, Val Loss:    0.2, Val Acc:  94.30%, Time: 0:04:39 
Iter:   1900, Train Loss:  0.031, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  93.82%, Time: 0:04:57 
Iter:   2000, Train Loss:  0.018, Train Acc: 100.00%, Val Loss:   0.25, Val Acc:  93.00%, Time: 0:05:13 
Iter:   2100, Train Loss:   0.01, Train Acc: 100.00%, Val Loss:   0.21, Val Acc:  93.62%, Time: 0:05:28 
Iter:   2200, Train Loss:  0.039, Train Acc:  98.44%, Val Loss:   0.22, Val Acc:  94.08%, Time: 0:05:44 
Iter:   2300, Train Loss:  0.071, Train Acc:  96.88%, Val Loss:   0.18, Val Acc:  94.92%, Time: 0:05:59 
Epoch: 4
Iter:   2400, Train Loss:  0.045, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  94.20%, Time: 0:06:15 
Iter:   2500, Train Loss:   0.04, Train Acc:  98.44%, Val Loss:    0.2, Val Acc:  94.82%, Time: 0:06:31 
No optimization for a long time, auto-stopping...

```

在验证集上的最佳效果为94.12%，且只经过了3轮迭代就已经停止。

准确率和误差如图所示：

![images](images/acc_loss.png)


### 测试

运行 `python3 run_cnn.py test` 在测试集上进行测试。

```
Test Loss:   0.13, Test Acc:  96.17%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          体育       1.00      0.99      0.99      1000
          财经       0.96      0.99      0.98      1000
          房产       1.00      1.00      1.00      1000
          家居       0.98      0.88      0.93      1000
          教育       0.92      0.92      0.92      1000
          科技       0.92      0.98      0.95      1000
          时尚       0.97      0.96      0.96      1000
          时政       0.96      0.93      0.95      1000
          游戏       0.96      0.98      0.97      1000
          娱乐       0.96      0.98      0.97      1000

    accuracy                           0.96     10000
   macro avg       0.96      0.96      0.96     10000
weighted avg       0.96      0.96      0.96     10000

Confusion Matrix...
[[989   0   1   0   4   2   0   2   2   0]
 [  0 993   0   0   2   1   0   4   0   0]
 [  0   0 996   1   2   1   0   0   0   0]
 [  1  15   2 883  25  26  10  21   7  10]
 [  0   8   1   8 920  26  10  10   9   8]
 [  0   0   0   1   3 980   4   1  11   0]
 [  1   0   0   6   9   2 962   0   6  14]
 [  0  16   1   0  26  17   0 934   1   5]
 [  0   2   0   2   3   2   4   1 983   3]
 [  1   0   0   3   7   4   4   1   3 977]]
Time usage: 0:00:10（原作者的时间是0：00：05）
```

在测试集上的准确率达到了96.17%(和原作者的不一样，原作者是96.04），且各类的precision, recall和f1-score都超过了0.9。

从混淆矩阵也可以看出分类效果非常优秀。

## RNN循环神经网络

### 配置项

RNN可配置的参数如下所示，在`rnn_model.py`中。

```python
class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
```

### RNN模型

具体参看`rnn_model.py`的实现。

大致结构如下：

![images/rnn_architecture](images/rnn_architecture.png)

### 训练与验证

> 这部分的代码与 run_cnn.py极为相似，只需要将模型和部分目录稍微修改。

运行 `python run_rnn.py train`，可以开始训练。

> 若之前进行过训练，请把tensorboard/textrnn删除，避免TensorBoard多次训练结果重叠。

```
#0724从这里开始
```

在验证集上的最佳效果为91.42%，经过了8轮迭代停止，速度相比CNN慢很多。

准确率和误差如图所示：

![images](images/acc_loss_rnn.png)


### 测试

运行 `python run_rnn.py test` 在测试集上进行测试。

```
Testing...
Test Loss:   0.21, Test Acc:  94.22%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         体育       0.99      0.99      0.99      1000
         财经       0.91      0.99      0.95      1000
         房产       1.00      1.00      1.00      1000
         家居       0.97      0.73      0.83      1000
         教育       0.91      0.92      0.91      1000
         科技       0.93      0.96      0.94      1000
         时尚       0.89      0.97      0.93      1000
         时政       0.93      0.93      0.93      1000
         游戏       0.95      0.97      0.96      1000
         娱乐       0.97      0.96      0.97      1000

avg / total       0.94      0.94      0.94     10000

Confusion Matrix...
[[988   0   0   0   4   0   2   0   5   1]
 [  0 990   1   1   1   1   0   6   0   0]
 [  0   2 996   1   1   0   0   0   0   0]
 [  2  71   1 731  51  20  88  28   3   5]
 [  1   3   0   7 918  23   4  31   9   4]
 [  1   3   0   3   0 964   3   5  21   0]
 [  1   0   1   7   1   3 972   0   6   9]
 [  0  16   0   0  22  26   0 931   2   3]
 [  2   3   0   0   2   2  12   0 972   7]
 [  0   3   1   1   7   3  11   5   9 960]]
Time usage: 0:00:33
```

在测试集上的准确率达到了94.22%，且各类的precision, recall和f1-score，除了家居这一类别，都超过了0.9。

从混淆矩阵可以看出分类效果非常优秀。

对比两个模型，可见RNN除了在家居分类的表现不是很理想，其他几个类别较CNN差别不大。

还可以通过进一步的调节参数，来达到更好的效果。


## 预测

为方便预测，repo 中 `predict.py` 提供了 CNN 模型的预测方法。
