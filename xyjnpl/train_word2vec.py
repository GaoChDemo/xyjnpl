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

# 加载语料并训练
sentences = word2vec.Text8Corpus(CONFIG.PATH_TRAIN_WORD2VEC)
model = word2vec.Word2Vec(sentences, sg=1, size=100, hs=1, min_count=1, window=3)
model.save(CONFIG.PATH_SAVE_WORD2VEC)
