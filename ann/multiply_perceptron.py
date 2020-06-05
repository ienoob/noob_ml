#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    多层感知机, 全连接神经网络
    标准的bp神经网络 == 随机梯度下降 stochastic gradient descent
    累积的bp神经网络 == 批梯度下降 batch gradient descent
"""
import numpy as np


class MultiplyPerceptron(object):

    def __init__(self):
        self.max_iter = 10000 # 迭代次数
        self.hidden = 10  # 隐藏层节点数
        self.output = 2  # 输出节点数
        self.learning_rate1 = 0.001  # 学习率1
        self.learning_rate2 = 0.001  # 学习率2

        self.w = None # 第一层
        self.v = None
        self.b1 = None
        self.b2 = None

    def fit(self, train, target):

        row, column = train.shape
        target = target.reshape((len(target), 1))
        target_2 = 1 - target
        target = np.hstack((target_2, target))

        w = np.random.randn(column, self.hidden)
        v = np.random.randn(self.hidden, self.output)
        b1 = np.random.randn(self.hidden, 1)
        b2 = np.random.randn(self.output, 1)

        z1 = np.dot(train, w) + np.tile(b1, (1, row)).T
        cita1 = sigmoid(z1)

        z2 = np.dot(cita1, v) + np.tile(b2, (1, row)).T
        cita2 = sigmoid(z2)
        cita2 = np.apply_along_axis(softmax, 1, cita2)
        # print(cita2)

        j = 0
        ex = 0
        while ex < self.max_iter:
            print(np.sum(np.square(cita2-target))/row/2)
            batch = 1
            delta_w_m = np.zeros((column, self.hidden))
            delta_v_m = np.zeros((self.hidden, self.output))
            delta_b1_m = np.zeros((self.hidden, 1))
            delta_b2_m = np.zeros((self.output, 1))

            for i in range(batch):
                i = j % row
                y = target[i]
                y_ = cita2[i]

                delta_y = y_ - y
                delta_f = np.eye(self.output, dtype=float)
                for j in range(self.output):
                    delta_f[j][j] = y_[j]*(1-y_[j])
                delta_y = delta_y.reshape((self.output, 1))
                last_y = cita1[i].reshape((1, self.hidden))

                delta_v = np.dot(np.dot(delta_f, delta_y), last_y).T
                delta_b2 = np.dot(delta_f, delta_y)

                delta_t = []
                for k in range(self.hidden):
                    p = sum(np.dot(v[k], delta_b2))
                    q = cita1[i][k] * (1-cita1[i][k])
                    delta_t.append(p*q)
                delta_t = np.array(delta_t).reshape((self.hidden, 1))
                x = train[i].reshape((1, column))

                delta_w = np.dot(delta_t, x).T
                delta_b1 = delta_t

                delta_w_m += delta_w
                delta_v_m += delta_v
                delta_b1_m += delta_b1
                delta_b2_m += delta_b2

                j += 1

            w = w - self.learning_rate1 * delta_w_m
            b1 = b1 - self.learning_rate1* delta_b1_m

            v = v - self.learning_rate2 * delta_v_m
            b2 = b2 - self.learning_rate2 * delta_b2_m

            z1 = np.dot(train, w) + np.tile(b1, (1, row)).T
            cita1 = sigmoid(z1)

            z2 = np.dot(cita1, v) + np.tile(b2, (1, row)).T
            cita2 = sigmoid(z2)

            ex += 1
            # break
        self.w = w
        self.v = v
        self.b1 = b1
        self.b2 = b2

    def predict(self, data):
        row, column = data.shape
        z1 = np.dot(data, self.w) + np.tile(self.b1, (1, row)).T
        cita1 = sigmoid(z1)

        z2 = np.dot(cita1, self.v) + np.tile(self.b2, (1, row)).T
        cita2 = sigmoid(z2)

        return cita2


def sigmoid(x):
    return 1.0/(1+np.exp(x))

def softmax(x):
    b = np.sum(np.exp(x))
    return np.exp(x)/b

if __name__ == "__main__":
    a = np.array([[1,1],[2,2]])
    print(np.apply_along_axis(softmax, 1, a))