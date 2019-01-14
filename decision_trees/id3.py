#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    id3算法
    一个比较基础的决策树算法
    使用information gain作为分裂树的依据
"""
import numpy as np


class TreeNode(object):

    def __init__(self):
        self.split_feature = None
        self.label = None
        self.sub_nodes = dict()


class ID3(object):

    def __init__(self):
        self.root = TreeNode()

    def fit(self, X, Y):
        """
        :param X: list<list<str>>
        :param Y: list<str>
        :return:
        """
        self.split(X, Y, self.root)

    def split(self, input_x, input_y, tree_node):
        entropy_y = entropy(input_y)
        max_info_gain = -1
        select_feature = -1
        feature_use = dict()

        # 计算信息增益
        for ie, f_values in enumerate(zip(*input_x)):
            feature_dict = {}
            for j, f_v in enumerate(f_values):
                feature_dict.setdefault(f_v, [])
                feature_dict[f_v].append(j)
            condition_entropy = 0.0
            for v in feature_dict.values():
                condition_y = [input_y[i] for i in v]
                condition_entropy += len(v)/len(f_values)*entropy(condition_y)
            entropy_gain = entropy_y-condition_entropy
            if max_info_gain < entropy_gain:
                max_info_gain = entropy_gain
                select_feature = ie
                feature_use = feature_dict

        # 设置分裂属性
        tree_node.split_feature = select_feature

        # 递归调用
        for k, v in feature_use.items():
            sub_input_x = [input_x[i] for i in v]
            sub_input_y = [input_y[i] for i in v]

            if is_same(sub_input_y):
                tree_node.sub_nodes[k] = sub_input_y[0]
            else:
                sub_node = TreeNode()
                self.split(sub_input_x, sub_input_y, sub_node)
                tree_node.sub_nodes[k] = sub_node

    def predict(self, input_x):
        node = self.root
        while isinstance(node, TreeNode):
            split_f = node.split_feature
            node = node.sub_nodes[input_x[split_f]]
        return node


# 检查数据集中元素是否都相同
def is_same(input_data):
    f = input_data[0]
    for x in input_data[1:]:
        if f != x:
            return False
    return True


# 计算数据熵
def entropy(input_data):
    type_dict = {}
    data_len = len(input_data)
    for data in input_data:
        type_dict.setdefault(data, 0)
        type_dict[data] += 1
    en_value = 0
    for v in type_dict.values():
        en_value += v/data_len*np.log2(data_len/v)
    return en_value
