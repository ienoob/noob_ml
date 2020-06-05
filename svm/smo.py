#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    svm 是一种以结构化风险最小化的机器学习算法，
    实现svm算法有很多，其中比较流行的方式是SMO(Sequential Minimal Optimization)
"""
from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt


data_path = "D:\code\git\\noob_ml\\testSet.txt"

with open(data_path, "r") as f:
    data = f.read()

data = data.split("\n")
data = [x.split("\t") for x in data][:-1]
data = np.array(data, dtype=np.float32)
x1 = data[:, 0]
x2 = data[:, 1]
label = data[:, 2]



def if_kkt(alpha, gxi, yi, c):
    if alpha == 0:
        if gxi * yi >= 1:
            return True
    elif alpha < c:
        if gxi * yi == 1:
            return True
    elif alpha == c:
        if gxi * yi <= 1:
            return True
    else:
        return False
    return False

def random_other(i, num):
    j = i
    while j == i:
        j = random.uniform(0, num)
    return j

def smo(x, y, kernel):
    max_iter = 100
    data_num = 100
    alpha = np.zeros(data_num)
    b = 0
    C = 0.5
    for iter in range(max_iter):
        print("iter {} start".format(iter))
        # 第一个变量
        for i in range(data_num):
            alpha1 = alpha[i]
            gxi = sum([alpha[k]*y[k]*kernel(x[k], x[i]) for k in range(data_num)]) + b
            E1 = gxi - y[i]
            if (alpha1>0 and (y[i]*gxi)> 1) or (alpha1<C and (y[i]*gxi<1)):

                Ei_list = []
                for j in range(data_num):
                    if i == j:
                        continue
                # j = random_other(i, data_num)
                    alpha2 = alpha[j]

                    if y[i] == y[j]:
                        L = max(0, alpha2+alpha1-C)
                        H = min(C, alpha2+alpha1)
                    else:
                        L = max(0, alpha2-alpha1)
                        H = min(C, C+alpha2-alpha1)

                    gxj = sum([alpha[k]*y[k]*kernel(x[k], x[j]) for k in range(data_num)]) + b
                    E2 = gxj - y[j]
                    eta = kernel(x[i], x[i]) + kernel(x[j], x[j]) - 2*kernel(x[i], x[j])

                    alpha2_new = alpha2 + y[j] * (E1-E2) / eta
                    if alpha2_new > H:
                        alpha2_new = H
                    elif alpha2_new < L:
                        alpha2_new = L
                    if abs(alpha2_new-alpha2) < 0.0001:
                        print("not enough moving")
                        continue
                    Ei_list.append((abs(E1-E2), j, alpha2_new, alpha2, E2))
                if len(Ei_list):
                    Ei_list.sort(key=lambda x: x[0])

                    ind = Ei_list[-1][1]
                    alpha2_new = Ei_list[-1][2]
                    alpha2 = Ei_list[-1][3]
                    E2 = Ei_list[-1][4]

                    alpha[ind] = alpha2_new
                    alpha[i] = alpha1 + y[i]*y[ind]*(alpha2-alpha2_new)

                    b1 = -E1 - y[i]*kernel(x[i], x[i])*(alpha[i]-alpha1) - y[ind]*kernel(x[i], x[ind])*\
                         (alpha[ind]-alpha2) + b
                    b2 = -E2 - y[i] * kernel(x[i], x[ind]) * (alpha[i] - alpha1) - y[ind] * kernel(x[ind], x[ind]) * (
                            alpha[ind] - alpha2) + b
                    if 0 < alpha[i] < C and 0 < alpha[ind] < C:
                        b = b1
                    else:
                        b = (b1+b2)/2
                    print("iter {} choice index {} as alpha2".format(iter, ind))
    w = np.dot(np.multiply(alpha, label), x)
    j = 0
    for i, a in enumerate(alpha):
        if a > 0:
            j = i
            break
    b = y[j] - sum([alpha[k]*y[k]*kernel(x[k], x[j]) for k in range(data_num)])

    return w, b







def linear_kernal(x1, x2):
    return np.dot(x1, x2)

w, b = smo(data[:,:2], label, kernel=linear_kernal)
x2_hat = -1*(w[0]*data[:, 0] + b)/w[1]

plt.title("lnn scatter")
plt.scatter(x1, x2, c=["r" if v > 0 else "b" for v in label ])
# plt.plot(x1, x2_hat)
plt.show()
