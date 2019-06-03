# -*- coding: utf-8 -*-

"""
程序入口
本项目用于2019年毕设，使用CNN对《西游记》小说中的人物关系进行识别
"""
import sys
import os

# 当前项目路径加入到环境变量中，让解析器能找到第一model的目录
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

import xyjnpl.openfile as of
import config.setting as CONFIG
import xyjnpl.cnn as cnn
import xyjnpl.utils as utils
import xyjnpl.train_word2vec as train_word2vec
import xyjnpl.preprocessing as pre


# with open('model/tokenizer' + str(CONFIG.VERSION) + '.pickle', 'rb') as f:
#     tokenizer = pickle.load(f)
# data_test, labels_test = cnn.deal_data(tokenizer, sents_test_deal[36598:], bags_test_deal[36598:])
# cnn.load_models(data_test, labels_test, bags_test_deal[36598:], tokenizer)

# cnn.split_data(data, labels, word_index)
# cnn.load_models(data, labels, word_index)


def train_demo():
    sent_train = of.read_txt_and_deal(CONFIG.PATH_TRAIN_SENT)
    bags_train = of.read_txt_and_deal(CONFIG.PATH_TRAIN_BAG)
    # sent_train, bags_train = pre.delete_line(sent_train, bags_train, 5000)

    sent_test = of.read_txt_and_deal(CONFIG.PATH_TEST_SENT)
    bags_test = of.read_txt_and_deal(CONFIG.PATH_TEST_BAG)
    # sent_test, bags_test = pre.delete_line(sent_test, bags_test, 3000)

    sent_train_deal = [x[3] for x in sent_train]
    bags_train_deal = [x[1] for x in bags_train]
    bags_train_deal = utils.standard_bags(bags_train_deal)

    sent_test_deal = [x[3] for x in sent_test]
    bags_test_deal = [x[1] for x in bags_test]
    bags_test_deal = utils.standard_bags(bags_test_deal)

    data, labels, tokenizer = cnn.fit_tokenizer(sent_train_deal[200000:], bags_train_deal[200000:])
    train_word2vec.word2vec_train(sent_train_deal)
    data_test, labels_test = cnn.deal_data(tokenizer, sent_test_deal[30000:], bags_test_deal[30000:])
    model = cnn.fit_model(data, labels, tokenizer)
    cnn.evaluate_model(model, data_test, labels_test)


def train_test():
    train = of.read_txt_and_deal(CONFIG.PATH_TRAIN)
    train = pre.hide_nr_demo(train)
    sent_train_deal = [x[2] for x in train]
    bags_train_deal = [x[3] for x in train]
    data, labels, tokenizer = cnn.fit_tokenizer(sent_train_deal, bags_train_deal)
    x_train, y_train, x_test, y_test = cnn.split_data(data, labels)
    # train_word2vec.word2vec_train(sent_train_deal)
    # data_test, labels_test = cnn.deal_data(tokenizer, x_test, y_test)
    model = cnn.fit_model(x_train, y_train, tokenizer)
    cnn.evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    train_test()
