#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
有些大别墅数据特征分布太异常，直接当做异常点去掉
"""
index = 0
out = []
for g in open('./ori_all_data.csv', encoding='utf-8'):
    g = g.strip()
    if index == 0:
        out.append(g)
        index += 1
        continue
    g_arr = g.split(",")
    if int(g_arr[-10]) > 4:
        continue
    else:
        out.append(g)
    index += 1
with open('./all_data.csv', 'w',encoding='utf-8') as f:
    for g in out:
        f.write(g + "\n")


