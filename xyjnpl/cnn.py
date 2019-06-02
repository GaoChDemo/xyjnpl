# -*- coding: utf-8 -*-

"""
cnn模型计算以及word2vec嵌入
"""
import pickle

import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import config.setting as CONFIG
from keras.layers import Dense, Input, Flatten, Dropout, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.models import load_model
import numpy as np
from xyjnpl.metrics import Metrics
import xyjnpl.openfile as of

VECTOR_DIR = 'wiki.zh.vector.bin'  # 词向量模型文件


# 训练tokenizer模型,获取向量
def fit_tokenizer(sents, bags):
    # 获取所有句子
    tokenizer = Tokenizer()
    # 首先使用已经以空格分割的句子对tokenizer模型进行训练。
    tokenizer.fit_on_texts(sents)
    # 使用词id替换句子中出现的每一个词
    sequences = tokenizer.texts_to_sequences(sents)
    # 查看词id
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=CONFIG.MAX_SEQUENCE_LENGTH)
    labels = to_categorical(bags)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data, labels, tokenizer


def deal_data(tokenizer, sents, bags):
    sequences = tokenizer.texts_to_sequences(sents)
    data = pad_sequences(sequences, maxlen=CONFIG.MAX_SEQUENCE_LENGTH)
    labels = to_categorical(bags)
    return data, labels


# 拆分训练集、验证集、测试集
def split_data(data, labels, tokenizer):
    p1 = int(len(data) * (1 - CONFIG.VALIDATION_SPLIT - CONFIG.TEST_SPLIT))
    p2 = int(len(data) * (1 - CONFIG.TEST_SPLIT))
    # 训练集
    x_train = data[:p1]
    y_train = labels[:p1]
    # 验证集
    x_val = data[p1:p2]
    y_val = labels[p1:p2]
    # 测试集
    x_test = data[p2:]
    y_test = labels[p2:]
    print('train docs: ' + str(len(x_train)) + ' ' + str(len(y_train)))
    print('val docs: ' + str(len(x_val)) + ' ' + str(len(y_val)))
    print('test docs: ' + str(len(x_test)) + ' ' + str(len(y_test)))


def fit_model(x_train, y_train, tokenizer, x_val=None, y_val=None):
    word_index = tokenizer.word_index

    word2vec_model = gensim.models.Word2Vec.load('./word2vec')
    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embedding_matrix = np.zeros((len(word_index) + 1, word2vec_model.vector_size))
    for word, i in word_index.items():
        try:
            embedding_vector = word2vec_model[word]
            embedding_matrix[i] = embedding_vector
        except:
            continue

    model = Sequential()
    # 使用Embedding层将每个词编码转换为词向量
    model.add(Embedding(len(word_index) + 1, CONFIG.EMBEDDING_DIM,weights=[embedding_matrix], input_length=CONFIG.MAX_SEQUENCE_LENGTH))
    model.add(Dropout(CONFIG.DROPOUT))
    model.add(Conv1D(CONFIG.FILTERS, CONFIG.KERNEL_SIZE, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(CONFIG.HIDDEN_DIMS, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # plot_model(model, to_file='model.png',show_shapes=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()

    print('####################################### 开始训练model #############################################')
    if x_val is not None and y_val is not None:
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
    else:
        model.fit(x_train, y_train, epochs=10, batch_size=128)
    with open('model/tokenizer' + str(CONFIG.VERSION) + '.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    model.save('model/word_vector_cnn_' + str(CONFIG.VERSION) + '.h5')
    return model


def evaluate_model(model, x_test, y_test):
    print('####################################### 开始验证model #############################################')
    y_predict = model.predict_classes(x_test)
    for x in range(len(x_test)):
        print(y_predict[x])
    print(model.evaluate(x_test, y_test))


def evaluate_model(model, x_test, y_test, bags_train_deal):
    print('####################################### 开始验证model #############################################')
    y_predict = model.predict_classes(x_test)
    for x in range(len(x_test)):
        print(y_predict[x], end=', ')
        print(bags_train_deal[x])
    me = Metrics()
    me.calculate(y_predict, bags_train_deal)
    print(model.evaluate(x_test, y_test))


# 读取模型进行处理
def load_models(data, labels, labelss, tokenizer):
    model = load_model('model/word_vector_cnn_' + str(CONFIG.VERSION) + '.h5')
    print('test docs: ' + str(len(data)) + ' ' + str(len(labels)))
    y_predict = model.predict_classes(data)
    for x in range(len(data)):
        # print(y_predict[i])
        if y_predict[x] != 0 and labelss[x] != 0:
            print(y_predict[x], end=',')
            print(labelss[x], end=',')
            print(str(y_predict[x]) == str(labelss[x]))
    print(model.evaluate(data, labels))
