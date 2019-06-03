# -*- coding: utf-8 -*-

"""
标准化西游记小说文档
"""

import re
import jieba
import xyjnpl.openfile as of
import xyjnpl.utils as utils

# 将西游记文档截取成句子保存文件
def cut_sentence(txt):
    sentences = re.split('。|\n', txt)
    result = list()
    for i in range(len(sentences)):
        sentence = sentences[i].replace(' ', '').replace('　', '')
        if sentence.startswith("”") or sentence.startswith("’") or sentence.startswith("“"):
            pop_str = result.pop()
            sentence = pop_str + sentence
        if len(sentence) > 2:
            result.append(sentence)

    return result


# 将西游记句子截取词成保存文件
def cut_word(nr_path, sentences):
    # 加载自定义nr词典
    jieba.load_userdict(nr_path)
    nr_list = of.read_list_and_deal(nr_path)
    result = list()
    no_result = list()
    nrs = list()
    for item in sentences:
        words = jieba.lcut(item)
        words, count, nr = mark_nr(words, nr_list)
        word_str = ''.join(words)
        if count >= 1:
            result.append(word_str)
        else:
            no_result.append(word_str)

    return result, no_result


def mark_nr(words, nr_list):
    count = 0
    nr = list()
    for i in range(len(words)):
        if words[i] in nr_list:
            nr.append(words[i])
            words[i] = '「' + words[i] + '」'
            count += 1
    return words, count, nr


# 标准化文档
def standard_test(sent_path, bag_path, path_train):
    sents_train = of.read_txt_and_deal(sent_path)
    bags_train_r = of.read_txt_and_deal(bag_path)
    bags_train = utils.standard_bags([i[1] for i in bags_train_r])
    for i in range(len(sents_train)):
        lis = sents_train[i][1:4].copy()
        lis.append(bags_train[i])
        of.write_list_line(lis, path_train % ('train'+bags_train[i]))
