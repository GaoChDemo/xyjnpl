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
KERNEL_SIZE = 3  # 卷积核的空域或时域窗长度
EMBEDDING_DIM = 50  # 词向量空间维度
FILTERS = 250  # 卷积核数
HIDDEN_DIMS = 250  # 第一次卷积输出大小
VALIDATION_SPLIT = 0.16  # 验证集比例
DROPOUT = 0.5  # Dropout
TEST_SPLIT = 0.2  # 测试集比例
VERSION = 151  # 测试
