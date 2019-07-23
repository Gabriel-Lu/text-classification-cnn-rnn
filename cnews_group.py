#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Function:将ThuCnews文件夹中的代表多个类目的多个文件整合到一个文件中，执行后得到三个数据文件
thucnews文件中有很多种类的很多条数据#需要自行在https://github.com/gaussic/text-classification-cnn-rnn下载
 cnews.train.txt 训练集50k条
 cnews.test.txt 验证集5k条
 cnews.val.txt 测试集10k条

Usage:对于法律项目而言，只需要：
    1）先去掉法律文书中的“判决如下...",只保留前面的其他内容
    2）按照事件顺序，匹配有期徒刑年限到1）中的文本，生成一个新文本
    3）按照train:test:validation=5：1：0.5生成三个文件
"""

import os

def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '') #删除掉了：换行;缩进;全角空白符

def save_file(dirname):#data/thucnews
    """
    将多个文件整合并存到3个文件中
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """
    f_train = open('data/cnews/cnews.train.txt', 'w', encoding='utf-8')
    f_test = open('data/cnews/cnews.test.txt', 'w', encoding='utf-8')
    f_val = open('data/cnews/cnews.val.txt', 'w', encoding='utf-8')
    for category in os.listdir(dirname):   # 分类目录
        print(category)
        cat_dir = os.path.join(dirname, category)#用于路径拼接，拼接文件路径
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)
            if count < 5000:#从每类中各写5k个到train set。一共有10类，所以train set含有5k*10条数据
                f_train.write(category + '\t' + content + '\n')
            elif count < 6000:#1k个到test.set
                f_test.write(category + '\t' + content + '\n')
            else:#rest（500）条写入validation test
                f_val.write(category + '\t' + content + '\n')
            count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()

if __name__ == '__main__':
    '''
    if _name_ == '_main_':的作用
    当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
    当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
    '''
    save_file('data/thucnews')#需要自行在https://github.com/gaussic/text-classification-cnn-rnn下载
    print(len(open('data/cnews/cnews.train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('data/cnews/cnews.test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('data/cnews/cnews.val.txt', 'r', encoding='utf-8').readlines()))


