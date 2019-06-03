# -*- coding: utf-8 -*-

"""
程序入口
本项目用于2019年毕设，使用CNN对《西游记》小说中的人物关系进行识别
"""

import sys

"""
0      NA     0 
1-6    配偶    1
7-24   血亲    2
25-29  姻亲    3
30     友谊关系 4
31-32  感情关系 5
33-34  师生关系 6
"""


def standard_bags(bags):
    res = []
    for itemStr in bags:
        item = int(itemStr)
        if item == 0:
            res.append('0')
        elif 1 <= item <= 6:
            res.append('1')
        elif 7 <= item <= 24:
            res.append('2')
        elif 25 <= item <= 29:
            res.append('6')
        elif 30 == item:
            res.append('4')
        elif 31 <= item <= 32:
            res.append('5')
        elif 33 <= item <= 34:
            res.append('3')
        else:
            print("no found " + str(item))
            sys.exit(0)
    return res


def one_hot_to_list(datas):
    res = list()
    for data in datas:
        i = 0
        for x in data:
            if x > 0:
                break
            else:
                i = i + 1
        res.append(i)
    return res
