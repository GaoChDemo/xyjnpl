# -*- coding: utf-8 -*-

"""
标准化西游记小说文档
"""

import re
import jieba
import xyjnpl.openfile as of


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
