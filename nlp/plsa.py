#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Probabilistic Latent Semantic Analysis
    实现PLSA模型
    参考 http://zhikaizhang.cn/2016/06/17/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B9%8BPLSA/
"""
# import os
import jieba
import numpy as np


class PLSA(object):

    def __init__(self, doc_list, topic_num=5):

        # 文档数量
        self.doc_num = 0
        # 词汇数量
        self.word_num = 0
        self.word_metrix = dict()
        self.word_map = dict()

        for i, doc in enumerate(doc_list):
            self.word_metrix.setdefault(i, {})
            self.doc_num += 1
            for w in doc:
                if w not in self.word_map:
                    self.word_map[w] = self.word_num
                    self.word_num += 1
                wi = self.word_map[w]
                self.word_metrix[i].setdefault(wi, 0)
                self.word_metrix[i][wi] += 1
        self.dw = np.zeros((self.doc_num, self.word_num))

        for k, wordmap in self.word_metrix.items():
            for i, num in wordmap.items():
                self.dw[k][i] = num

        # 主题数量
        self.topic_num = topic_num

        # 随机生成p(z/d)
        self.dz = np.random.rand(self.doc_num, self.topic_num)
        self.dz = self.dz/self.dz.sum(axis=1).reshape((self.doc_num, 1))

        # 随机生成p(w/z)
        self.zw = np.random.rand(self.topic_num, self.word_num)
        self.zw = self.zw/self.zw.sum(axis=1).reshape((self.topic_num, 1))

        self.zdw = np.zeros((self.doc_num, self.word_num, self.topic_num))

        # 最大迭代次数
        self.max_iter = 100

        print(self.dz)
        print(self.zw)
        print(self.dw.shape)

    def em(self):
        for _ in range(self.max_iter):
            # E 步
            for i in range(self.doc_num):
                for j in range(self.word_num):
                    for k in range(self.topic_num):
                        self.zdw[i][j][k] = self.dz[i][k]*self.zw[k][j] / np.sum([self.dz[i][ki]*self.zw[ki][j] for ki in range(self.topic_num)])

            # M 步
            for i in range(self.doc_num):
                for j in range(self.word_num):
                    ndi = np.sum([self.zdw[i][mj][ki] * self.dw[i][mj] for ki in range(self.topic_num) for mj in
                                            range(self.word_num)])
                    for k in range(self.topic_num):
                        self.dz[i][k] = np.sum([self.zdw[i][mj][k]*self.dw[i][mj] for mj in range(self.word_num)])/ndi

                        self.zw[k][j] = np.sum([self.zdw[ni][j][k]*self.dw[ni][j] for ni in range(self.doc_num)])/\
                                        np.sum([self.zdw[ni][mj][k]*self.dw[ni][mj] for ni in range(self.doc_num) for mj in range(self.word_num)])
                        # print(self.dz[i][k], self.zw[k][j])
            # print(self.dz)


if __name__ == "__main__":
    stop_set = {"\u3000", "\n", " ", "：", "，", "”", "“", ".", "。", "（", "）", "《", "》", "、"}
    doc_path = "D:\code\git\\noob_tf\C000008"
    # list_dir = os.listdir(doc_path)
    list_dir = ["10.txt", "11.txt", "12.txt", "13.txt", "14.txt", "15.txt", "16.txt"]
    doc_list = []
    for i, file in enumerate(list_dir):
        file_path = doc_path + "\\" + file
        with open(file_path, "r") as f:
            content = f.read()
        words_list = [x for x in jieba.cut(content) if x not in stop_set]
        print(file)
        print(words_list)
        doc_list.append(words_list)
    plsa = PLSA(doc_list, 6)
    plsa.em()



