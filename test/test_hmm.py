#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from nlp.hmm import HMM

"""
    测试隐马尔科夫模型，使用统计学习方法上的例子
"""
# 观察集合
OB = ["红", "白"]
# 状态转移矩阵
A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
# 状态转换观测结果矩阵
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
# 初始状态值
Pi = np.array([[0.2], [0.4], [0.4]])

# 观察序列
V = [0, 1, 0]

# 测试计算观测序列概率
true_result = 0.13022

hmm = HMM(A, B, Pi)

# 前向
forward_result = hmm.forward(V)
print("forward method result is {}".format(forward_result))
variance = np.abs(forward_result-true_result)
assert variance < 0.00001

# 后向
backward_result = hmm.backward(V)
print("backward method result is {}".format(backward_result))
variance = np.abs(backward_result-true_result)
assert variance < 0.00001

# 测试维特比算法， viterbi
true_pb = 0.0147
true_state = [2, 2, 2]

pre_pb, pre_state = hmm.viterbi(V)
print("viterbi max probability is {}".format(pre_pb))
print("viterbi max state path is {}".format(pre_state))
variance1 = np.abs(pre_pb-true_pb)
assert variance < 0.00001
assert pre_state == true_state
