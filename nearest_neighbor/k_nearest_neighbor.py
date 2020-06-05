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
        self.kdt = KDTree()

    def fit(self, train, target):
        self.train = train
        self.target = target

    # 使用kd 树
    def fit_usekd(self, train, target):
        input_data = [x+[target[i]] for i, x in enumerate(train)]
        self.kdt.build(input_data, len(train[0]))

    # 使用kd树预测
    def _predict_kd(self, data):
        predict_result = []
        for da in data:
            res = self.kdt.search_knn(da, self.k)
            knn_label = self.majority_vote([x[1] for x in res])
            predict_result.append(knn_label)
        return predict_result

    def predict(self, data, p_type=None):
        """
        :param data: List<List<Double>>
        :param p_type
        :return:
        """
        if p_type == "kdtree":
            return self._predict_kd(data)
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

        index_result = list(map(lambda x: x[0], distance_list))

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

    @staticmethod
    def majority_vote(da):
        """
        :param da: List<Double>
        :return:
        """
        target_statis = {}
        for d in da:
            target_statis.setdefault(d, 0)
            target_statis[d] += 1
        target_list = [(k, v) for k, v in target_statis.items()]
        target_list = sorted(target_list, key=lambda x: x[1], reverse=True)

        return target_list[0][0]


class Tree(object):

    def __init__(self):
        pass


class Node(object):

    def __init__(self):
        self.index = None
        self.x = None
        self.parent = None
        self.left = None
        self.right = None
        self.split_feature = None
        self.label = None


class KDTree(object):

    def __init__(self):
        self.root = Node()
        self.index = 1

    def _build(self, data_list, node, f_len, r=0, p=None):
        if len(data_list) == 1:
            node.x = data_list[0]
        elif len(data_list) > 0:
            data_list.sort(key=lambda xi: xi[r])
            mid_index = len(data_list)//2
            x = data_list[mid_index]
            node.x = x
            node.split_feature = r
            node.left = self._build(data_list[:mid_index], Node(), f_len, (r+1) % f_len, node)
            node.right = self._build(data_list[mid_index+1:], Node(), f_len, (r+1) % f_len, node)
        else:
            return None

        # 标记上序号id
        node.index = self.index
        self.index += 1

        # 默认输入数据最后一位是标签
        if f_len < len(node.x):
            node.label = node.x[-1]
        if p:
            node.parent = p
        return node

    def build(self, data_list, f_len=None):
        f_len = len(data_list[0]) if f_len is None else f_len
        self.root = self._build(data_list, Node(), f_len, 0)

    # 欧几里得距离
    @staticmethod
    def get_distance(a, b):
        """
        :param a:
        :param b:
        :return:
        """
        c = zip(a, b)
        d = map(lambda x: (x[0] - x[1]) ** 2, c)
        return math.sqrt(sum(d))

    def search_knn(self, x, k=1):
        if k < 1:
            raise Exception("k should bigger than 0")
        top_k = []
        search_set = set()
        node = self.root

        while node and node.split_feature:
            if x[node.split_feature] > node.x[node.split_feature]:
                if node.right is None:
                    break
                node = node.right
            else:
                if node.left is None:
                    break
                node = node.left

        while len(search_set) < self.index:
            if node.index not in search_set:
                if len(top_k) < k:
                    top_k.append((self.get_distance(x, node.x), node.label))

                else:
                    mk = self.get_distance(node.x, x)
                    if mk < top_k[-1][0]:
                        top_k[-1] = (mk, node.label)
                search_set.add(node.index)
                top_k.sort(key=lambda xi: xi[0])

            if node.left  or node.right:
                max_v = top_k[-1][0]
                sphree_size = abs(x[node.split_feature]-node.x[node.split_feature])
                if max_v > sphree_size:
                    if node.left:
                        node = node.left
                    else:
                        node = node.right
                else:
                    if node.index == 1:
                        break
                    else:
                        node = node.parent
            else:
                if node.index == 1:
                    break
                else:
                    node = node.parent

        return top_k

    def tobottom(self):
        pass
