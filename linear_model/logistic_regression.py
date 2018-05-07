#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    logistic regression 实现
"""
import numpy as np


class LogisticRegression(object):

    def __init__(self, max_iter=1000):
        self.weight = None # 初始权重
        self.max_iter = max_iter # 最大迭代次数
        self.min_err = 0.0001 # 最小变化
        self.alpha = 0.01 # 学习率

    def fit(self, train, target):
        """
        :param train: numpy.ndarray
        :param target: numpy.ndarray
        :return:
        """
        row, colum = train.shape
        w_size = colum + 1
        self.weight = np.zeros((w_size, 1))
        train = np.hstack((train, np.ones((row, 1))))
        target = target.reshape((row, 1))

        i = 0
        m = np.iinfo(np.int32).max
        last_w = self.weight
        while i < self.max_iter and m > self.min_err:
            delta_y = sigmoid(np.dot(train, self.weight)) - target
            delta_w = np.dot(train.T, delta_y)

            self.weight = self.weight - self.alpha*delta_w
            m = np.max(np.abs(last_w - self.weight))
            last_w = self.weight
            i += 1

        print("training work is done!")

    def predict(self, data):
        data = np.hstack((data, np.ones((data.shape[0], 1))))
        predict = sigmoid(np.dot(data, self.weight))
        predict[predict>0.5] = 1
        predict[predict<=0.5] = 0
        return predict


def get_label(x, confid=0.5):
    if x >= confid:
        return 1
    else:
        return 0




def gradient_decent():
    pass

def sigmoid(x):
    return 1.0/(1+np.exp(-x))


if __name__ == "__main__":
    print(np.random.randn(10).reshape((10, 1)))
