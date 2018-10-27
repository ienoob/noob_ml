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







# 前向后向方法

