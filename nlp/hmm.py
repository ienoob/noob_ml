#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    隐马尔科夫模型
    三个问题要解决
    1 计算概率
    2 预测问题
    3 估计参数
"""
import numpy as np


class HMM(object):

    def __init__(self, big_a, big_b, pi):
        # 状态转移矩阵
        self.big_a = big_a
        # 观测矩阵
        self.big_b = big_b
        # 初始状态
        self.pi = pi

    # 前向算法
    def forward(self, big_v):
        pb_ = (self.pi*self.big_b[:, big_v[0]:big_v[0]+1])
        for ob in big_v[1:]:
            transfer_state = np.dot(pb_.T, self.big_a)
            pb_ = transfer_state.T*self.big_b[:, ob:ob+1]
        return pb_.sum()

    # 后向算法
    def backward(self, big_v):
        back_ = np.ones((self.pi.shape[0], 1))
        for ob in big_v[:0:-1]:
            back_ = np.dot(self.big_a, back_*self.big_b[:, ob:ob+1])

        back_ = back_*self.big_b[:, big_v[0]:big_v[0]+1]*self.pi
        return back_.sum()

    # viterbi
    def viterbi(self, big_v):
        _state = []
        pb_ = self.pi*self.big_b[:, big_v[0]:big_v[0]+1]
        for ob in big_v[1:]:
            pb_matrix = pb_ * self.big_a
            pb_ = np.max(pb_matrix, axis=0)
            max_index = np.argmax(pb_matrix, axis=0)
            _state.append(max_index)
            pb_ = pb_.reshape((pb_.shape[0], 1))
            pb_ = pb_ * self.big_b[:, ob:ob+1]
        _state = np.array(_state).T
        max_pb = np.max(pb_)
        max_index = np.argmax(pb_, axis=0)[0]
        max_state = [max_index]
        for i in range(len(big_v)-2, -1, -1):
            max_index = _state[max_index][i]
            max_state.append(max_index)
        return max_pb, max_state
