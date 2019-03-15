#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt


def beta_s(x, a, b):
    return x**(a-1)*(1-x)**(b-1)
def beta(x, a, b):
    return beta_s(x, a, b)/ss.beta(a, b)

a = 0.1
b = 0.1
cur = np.random.rand()
states = [cur]
for i in range(10 ** 5):
    next = u = np.random.rand()
    if u < np.min((beta_s(next, a, b) / beta_s(cur, a, b), 1)):
        states.append(next)
        cur = next
    print(next)