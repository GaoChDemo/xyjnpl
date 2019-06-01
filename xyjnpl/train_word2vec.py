# -*- coding: utf-8 -*-

"""
训练word2vec模型并保存
"""
import sys
import os

# 当前项目路径加入到环境变量中，让解析器能找到第一model的目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

from gensim.models import word2vec
import config.setting as CONFIG
import re


def word2vec_train(data):
    print("start generate word2vec model...")
    seq = re.compile(' ')
    sentence_matrix = list()
    for line in data:
        line_seq = seq.split(line.strip())
        sentence_matrix.append(line_seq)
    # sentence_matrix = word2vec.Text8Corpus("cuttext_all_large.txt")
    # sentence_matrix = of.read_txt_and_deal_spa("cuttext_all_large.txt")

    model = word2vec.Word2Vec(sentence_matrix, size=CONFIG.EMBEDDING_DIM)  # 默认size=100 ,100维
    model.save('word2vec')
    print('finished and saved!')
    return model
