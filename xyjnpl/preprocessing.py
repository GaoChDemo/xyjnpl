# -*- coding: utf-8 -*-

"""
预处理文件
"""

import jieba
import jieba.posseg as pseg
import xyjnpl.openfile as of
import re


# 获取nr词
def query_name(path):
    txt = of.read_file_txt(path)
    excludes = {"来到", "一个", "国王", "我们", "变成", "你们", "什么"}
    for ch in '!@#$%^&*()_+-=[]\{}|;:,./;:；："，。<>?':
        txt = txt.replace(ch, "")
    words = pseg.cut(txt)
    counts = {}
    for w in words:
        if len(w.word) == 1:
            continue
        if w.flag == 'nr':
            counts[w.word] = counts.get(w.word, 0) + 1
    keys_list = list(counts.keys())
    for word in excludes:
        if word in keys_list:
            del counts[word]
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)
    return items


# 检测数据格式
def check_txt_column_number(data, index):
    for line in data:
        if len(line) != index:
            return False
    return True


# 获取全篇单词总数 用于demo
def query_word_count(data, index):
    seq = re.compile(' ')
    counts = set()
    for line in data:
        words = seq.split(line[index].strip())
        for w in words:
            counts.add(w)
    return counts


# 文本分词 获取人物出现频率
def txt_cut(txt):
    excludes = {"来到", "一个", "国王", "我们", "变成", "你们", "什么"}
    for ch in '!@#$%^&*()_+-=[]\{}|;:,./;:；："，。<>?':
        txt = txt.replace(ch, "")
    words = jieba.lcut(txt)
    counts = {}
    for word in words:
        if len(word) == 1:
            continue
        counts[word] = counts.get(word, 0) + 1
    for word in excludes:
        del counts[word]
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)
    return items


# 查询人物出场次数
def query_name_times(path):
    txt = of.read_file_txt(path)
    items = txt_cut(txt)
    while 1:
        num = eval(input("输入需要获得多少个出场次数最多的人物（3—8）:"))
        if 3 <= num <= 8:
            for i in range(num):
                word, count = items[i]
                print("{0:<10}{1:>5}".format(word, count))
            break
        else:
            continue


# 查看nr词典出现人物频率
def query_name_times_udf(nr_path, txt):
    # 加载自定义nr词典
    jieba.load_userdict(nr_path)
    nr_list = of.read_list_and_deal(nr_path)
    words = jieba.lcut(txt)
    counts = {}
    for word in words:
        if word in nr_list:
            counts[word] = counts.get(word, 0) + 1
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)
    return items


# 隐藏「」符号
def hide_nr(data, nr1, nr2):
    pattern = re.compile(r'「.*?」')
    out = re.sub(pattern, '*', data)
    out = re.sub(nr1, '*', out)
    out = re.sub(nr2, '*', out)
    return out
