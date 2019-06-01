# -*- coding: utf-8 -*-
if __name__ != '__main__':
    raise ImportError('This is not a module, but a script.')

import xyjnpl.openfile as of
import xyjnpl.preprocessing as pre
import xyjnpl.cnn as cnn
import xyjnpl.standardtxt as standard
import jieba

path_sent = '../../source/ccks2019/train/sent_train.txt'
path_bag = '../../source/ccks2019/train/sent_relation_train.txt'
path_xyj = '../../source/xyj.txt'
path_rw_me = '../../source/rw_me.txt'
path_rw = '../../source/nr.txt'
path_rw_deal = '../../source/nr_deal.txt'
path_dif = '../../source/dif.txt'
path_sentences = '../../source/sentences.txt'
path_words = '../../source/words.txt'
path_words_test = '../../source/words_test.txt'
path_no_words_test = '../../source/no_words_test.txt'
path_words_deal = '../../source/words_deal.txt'


def test_read_file_txt():
    print(of.read_file_txt(path_sent))


def test_read_txt_and_deal():
    print(of.read_txt_and_deal(path_sent))


def test_check_txt_column_number():
    print(pre.check_txt_column_number(of.read_txt_and_deal(path_sent), 4))


def test_query_word_count():
    words = pre.query_word_count(of.read_txt_and_deal(path_sent), 3)
    print(len(words))


def test_fit_tokenizer():
    sents = of.read_txt_and_deal(path_sent)
    bags = of.read_txt_and_deal(path_bag)
    sents_deal = [x[3] for x in sents]
    bags_deal = [x[1] for x in bags]
    model = cnn.fit_model(sents_deal, bags_deal)
    # cnn.evaluate_model(model,)
    # data, labels, word_index = cnn.fit_tokenizer(sents_deal, bags_deal)
    # cnn.split_data(data, labels, word_index)


def test_query_name_times():
    print(pre.query_name_times(path_xyj))


def test_query_name():
    of.write_dict(pre.query_name(path_xyj), path_rw_me)


def test_read_dict_and_deal():
    dic = of.read_dict_and_deal(path_rw_me)
    lis = of.read_list_and_deal(path_rw)
    rw_me = set(dic.keys())
    rw = set(lis)
    print(rw_me)
    print(rw)
    print(rw_me.intersection(rw))
    dif = rw_me.difference(rw)
    print(len(dif))
    print(dif)
    of.write_list(dif, path_dif)


def test_query_name_times_udf():
    txt = of.read_file_txt(path_xyj)
    items = pre.query_name_times_udf(path_rw, txt)
    for i in range(len(items)):
        word, count = items[i]
        print("{0:<10}{1:>5}".format(word, count))


def test_cut_sentence():
    txt = of.read_file_txt(path_xyj)
    sentence = standard.cut_sentence(txt)
    of.write_list(sentence, path_sentences)


def test_cut_word():
    txt = of.read_sentences(path_sentences)
    words, no_result = standard.cut_word(path_rw_deal, txt)
    of.write_list(words, path_words_test)
    of.write_list(no_result, path_no_words_test)


def test_demo():
    dataa = of.read_txt_and_deal(path_words_deal)
    sents_train_deal = list()
    for s in dataa:
        nr1 = s[0]
        nr2 = s[1]
        x = s[2]
        x = pre.hide_nr(x, nr1, nr2)
        words = jieba.lcut(x)
        # word_str = ' '.join(words)
        sents_train_deal.append(words)

    sents_train_deal.append(jieba.lcut('胡挺和胡磊结婚了'))
    sents_train_deal.append(jieba.lcut('摩拜单车被美团收购了，由美图经营'))

    for s in dataa:
        if len(s) != 4:
            print(s)

    bags_train_deal = [x[3] for x in dataa]
    bags_train_deal.append(1)
    bags_train_deal.append(0)

    data, labels, tokenizer = cnn.fit_tokenizer(sents_train_deal, bags_train_deal)
    data_test, labels_test = cnn.deal_data(tokenizer, sents_train_deal, bags_train_deal)
    model = cnn.fit_model(data[:-2], labels[:-2], tokenizer)
    cnn.evaluate_model(model, data_test, labels_test, bags_train_deal)


# test_cut_word()
# test_demo()
from xyjnpl.metrics import Metrics

me = Metrics()
lista = [1, 1, 1, 1, 2, 2, 2, 3, 3, 0, 0, 2]
listb = [1, 1, 2, 3, 2, 2, 3, 2, 3, 0, 2, 1]
val_predict = [x + 1 for x in lista]
val_tar = [x + 1 for x in listb]
me.calculate(val_predict, val_tar)
