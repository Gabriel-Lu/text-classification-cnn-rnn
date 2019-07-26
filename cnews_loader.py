# coding: utf-8
'''数据的预处理文件'''
from collections import Counter

import numpy as np #Numpy for 多维数组和矩阵计算
import tensorflow.keras as kr #Keras是支持快速开发的神经网络api


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    return word


def native_content(content):
    return content


def open_file(filename, mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """
    filename:
        待划分成一个一个字符的 label \t content形式的文本
    Function:
        返回contents：每一行作为一个列表，把自然文字分成一个一个的字符
        labels：把原标签也分成了一个一个的字符
    """
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')#strip（）删除空白字符;split('\t')按\t分割字符串
                if content:
                    # append函数在列表末尾添加新的对象#list把元祖转化成列表
                    contents=content.split( )  # 被告人 弭 某某

                    #contents.append(list(native_content(content)))  # append方法在列表末尾添加新的对象
                    #contents.append(native_content(content))
                    labels.append(native_content(label))
            except:
                pass

    #print("Contents")  # Test
    #print(contents)  # '被告人', '弭', '某某',  # Test
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """
    Function:
    构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理
        根据训练集构建词汇表（其实是词汇表），存储
    Input:
        train_dir:训练集的文件
    Output:
        vocab_dir:按出现次数递减排列的“词汇”表
    """
    data_train, _ = read_file(train_dir)  # data_train对应train_dir中的contents(contents是一个list,_对应train_dir中的label
    # 分成了字级别后的文字流被赋值给了data_train, 分成字后的标签被赋值给了_
    #print(data_train)#被告人 吴永飞 在 公路 上
    all_data = []  # 元组允许重复的元素
    '''将所有词汇加入词汇表，可能会存在重复的词汇'''
    for word in data_train:  # 对于每一个词语
        #print(word)#规定
        all_data.extend([word])  # all_data现在存储了在content中出现过的所有的字
    #print(all_data)  # '被告人', '弭', '某某'

    counter = Counter(all_data)#counter用于计算每个字一共出现过多少次
    count_pairs = counter.most_common(vocab_size - 1)   # 返回词汇表中出现次数最多的前vocab_size-1的词汇（字），以{“字”，“次数”}的格式存储
    words, _ = list(zip(*count_pairs))  # 按照出现频率将词汇存入词典中 #list(zip(*)))将元祖解压成列表#此时，words是一个list, 内容是出现次数最多的字
    '''添加一个 <PAD> 来将所有文本pad为同一长度'''
    #print(words)#Test
    words = ['<PAD>'] + list(words)
    '''我觉得这个words可以不要'''

    print(words)#test
    open_file(vocab_dir, mode='a').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表，返回词汇表和每个词汇对应的序号"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]#strip()删除空白
    word_to_id = dict(zip(words, range(len(words))))#zip（）打包成元组
    #print(word_to_id)#test？？？
    return words, word_to_id#词：id


def read_category():
    """读取分类目录，返回分类目录和每类对应的序号"""
    '''For Law,要修改成每半年一个类别'''
    categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    '''把被分成一个一个字的content中的内容连接成一个字符串words'''
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将数据集从文字转换为固定长度的id序列表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])     # 只把存在于词典中的词替换成了序号，否则直接抛弃
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad     # x_pad中每个元素是由序号代替文字的每一行的列表，y_pad是one-hot形式的标签


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


build_vocab('/home/tabrielluunn/200gb/text-classification-cnn-rnn-master/data/cnews/cnews.train.txt', '/home/tabrielluunn/200gb/text-classification-cnn-rnn-master/data/cnews/cnews.vocab.txt', vocab_size=500)

