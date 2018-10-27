#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from nlp.hmm import HMM

A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
Pi = np.array([[0.2], [0.4], [0.4]])

# 观察序列
V = [0, 1, 0]
true_result = 0.13022

hmm = HMM(A, B, Pi)
forward_result = hmm.forward(V)
print("forward method is {}".format(forward_result))
variance = np.abs(forward_result-true_result)
assert variance < 0.00001

backward_result = hmm.backward(V)
print("backward method is {}".format(backward_result))
variance = np.abs(backward_result-true_result)
assert variance < 0.00001

