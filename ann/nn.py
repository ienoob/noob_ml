#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

"""
    全连接神经网络
"""


class NN(object):

    # 定义三层结构
    def __init__(self, alpha=0.8):
        # 输入层
        self.input = None
        # 隐藏层
        self.hidden = None
        # 输出层
        self.output = None
        self.iternum = 10000
        # 学习率
        self.alpha = alpha

    def _init_weight(self):
        self.w = np.ones((self.input, self.hidden))
        self.bw = np.zeros((1, self.hidden))
        self.v = np.ones((self.hidden, self.output))
        self.bv = np.zeros((1, self.output))

    def fit(self, feature, target):
        data_num = feature.shape[0]

        self.input = feature.shape[-1]
        self.output = np.unique(target).shape[0]
        self.hidden = self.input * 1
        print(self.input, self.output, self.hidden)
        label = np.zeros((data_num, self.output))
        self._init_weight()

        for i, lb in enumerate(target):
            label[i][lb] = 1
        print(self.loss(feature, label))
        for _ in range(self.iternum):
            rd = self.random_choice(data_num)
            rd_feature, rd_label = feature[rd], label[rd]
            hidden, output = self.forward(feature, rd)

            for j in range(self.hidden):
                for k in range(self.output):
                    self.v[j][k] -= self.alpha*(output[k]-rd_label[k])*(1-output[k])*output[k]*hidden[j]
                    self.bv[0][k] -= self.alpha*(output[k]-rd_label[k])*(1-output[k])*output[k]

            for i in range(self.input):
                for j in range(self.hidden):
                    self.w[i][j] -= self.alpha*(1-hidden[j])*hidden[j]*rd_feature[i]\
                                    * np.sum([(output[k]-rd_label[k])
                                             * (1-output[k])*output[k]*self.v[j][k] for k in range(self.output)])
                    self.bw[0][j] -= self.alpha*(1-hidden[j])*hidden[j] \
                                     * np.sum([(output[k]-rd_label[k])*(1-output[k])
                                              * output[k]*self.v[j][k] for k in range(self.output)])

            print(self.loss(feature, label))

    # 前向传播
    def forward(self, feature, rd=None):
        hidden_ = np.dot(feature, self.w) + self.bw
        hidden = self.sigmoid(hidden_)

        output_ = np.dot(hidden, self.v) + self.bv
        output = self.sigmoid(output_)

        if rd is not None:
            return hidden[rd], output[rd]
        return hidden, output

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def loss(self, feature, label):
        _, output = self.forward(feature)
        return np.sum(np.square(output-label))/feature.shape[0]/label.shape[1]

    @staticmethod
    def random_choice(num):
        return np.random.randint(0, num)


if __name__ == "__main__":
    pass
