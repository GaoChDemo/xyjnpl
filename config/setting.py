# -*- coding: utf-8 -*-

"""
settings
"""

PATH_TRAIN_SENT = '../source/ccks2019/train/sent_train.txt'
PATH_TRAIN_BAG = '../source/ccks2019/train/sent_relation_train.txt'
PATH_TRAIN_WORD2VEC = '../source/word2vec/vector_word.txt'
PATH_SAVE_WORD2VEC = 'model/word2vec_model.model'
PATH_TEST_SENT = '../source/ccks2019/dev/sent_dev.txt'
PATH_TEST_BAG = '../source/ccks2019/dev/sent_relation_dev.txt'
MAX_SEQUENCE_LENGTH = 100  # 每条新闻最大长度
EMBEDDING_DIM = 150  # 词向量空间维度
VALIDATION_SPLIT = 0.16  # 验证集比例
TEST_SPLIT = 0.2  # 测试集比例
VERSION = 151 #测试
