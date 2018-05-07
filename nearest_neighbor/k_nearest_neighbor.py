#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    最近邻
"""
import math


class KNN(object):

    def __init__(self, k=1):
        self.k = k
        # 距离使用欧几里得距离
        self.distance = "Euclid"
        self.train = None
        self.target = None

    def fit(self, train, target):
        self.train = train
        self.target = target

    def predict(self, data):
        """
        :param data: List<List<Double>>
        :return:
        """
        predict_result = []
        for da in data:
            nearest_index = self.nearest_data(da)
            label_ = [self.target[i] for i in nearest_index]
            result = self.majority_vote(label_)
            predict_result.append(result)

        return predict_result

    def nearest_data(self, da):
        distance_list = []
        for i, d in enumerate(self.train):
            distance_list.append((i, self.get_distance(da, d)))

        distance_list.sort(key=lambda x: x[1])

        index_result = map(lambda x: x[0], distance_list)

        return index_result[:self.k]

    # 欧几里得距离
    @staticmethod
    def get_distance(a, b):
        """
        :param a:
        :param b:
        :return:
        """
        c = zip(a, b)
        d = map(lambda x: (x[0]-x[1])**2, c)
        return math.sqrt(sum(d))

    def majority_vote(self, da):
        """
        :param da: List<Double>
        :return:
        """
        target_statis = {}
        for d in da:
            target_statis.setdefault(d, 0)
            target_statis[d] += 1
        target_list = [(k, v) for k, v in target_statis.iteritems()]
        target_list = sorted(target_list, key=lambda x: x[1], reverse=True)

        return target_list[0][0]

    def kd_tree(self):
        pass




