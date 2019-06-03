# -*- coding: utf-8 -*-

"""
读取文件系列函数
"""

import sys
import re
import numpy as np


# 读取文件
def read_file_txt(path):
    out = ''
    try:
        out = open(path, 'r')
        print('open file:' + path)
        return out.read()
    except FileNotFoundError as e:
        print(e)
        sys.exit(0)
    finally:
        out.close()


# 获取文件并转换成数据 用于demo
def read_txt_and_deal(path):
    seq = re.compile('\t')
    data = list()
    with open(path, 'r') as f:
        for line in f:
            line_seq = seq.split(line.strip())
            data.append(line_seq)
    return data


# 获取文件并转换成数据 用于demo
def read_txt_and_deal_spa(path):
    seq = re.compile(' ')
    data = list()
    try:
        with open(path, 'r') as f:
            for line in f:
                line_seq = seq.split(line.strip())
                data.append(line_seq)
        return data
    except Exception as e:
        print(e)
        sys.exit(0)


def write_dict(dic, path):
    with open(path, 'w') as f:
        for k, y in dic:
            f.write(k + ',' + str(y) + '\n')


def write_list(lis, path):
    with open(path, 'w') as f:
        for x in lis:
            f.write(x + '\n')


def write_list_line(lis, path):
    with open(path, 'a+') as f:
        for x in lis:
            f.write(x)
            if x != lis[len(lis) - 1]:
                f.write('\t')
            else:
                f.write('\n')


def write_list_all_line(alllis, path):
    with open(path, 'a+') as f:
        for line in alllis:
            for x in line:
                f.write(x)
                if x != line[len(line) - 1]:
                    f.write('\t')
                else:
                    f.write('\n')


# 获取文件并转换成数据 用于demo
def read_dict_and_deal(path):
    seq = re.compile(',')
    data = dict()
    try:
        with open(path, 'r') as f:
            for line in f:
                line_seq = seq.split(line.strip())
                data[line_seq[0]] = line_seq[1]
        return data
    except Exception as e:
        print(e)
        sys.exit(0)


# 获取文件并转换成数据 用于demo
def read_list_and_deal(path):
    data = list()
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.replace('\n', '').split(' ')
                data.append(line[0])
        return data
    except Exception as e:
        print(e)
        sys.exit(0)


# 获取文件并转换成数据 用于demo
def read_sentences(path):
    data = list()
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                data.append(line)
        return data
    except Exception as e:
        print(e)
        sys.exit(0)


def get_training_word2vec_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]
