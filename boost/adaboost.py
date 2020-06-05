#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

class AdaBoost(object):

    def __init__(self):
        self.clf_num = 10

    def fit(self, train, target):
        m = train.shape[0]
        w = 1/m * np.ones(m)


        for i in range(4):
            ts = TreeStump()
            ts.fit(data, target, w)
            v = ts.adop_v
            err = ts.error
            ts_result = ts.result
            alpha = 1/2*np.log((1-err)/err)

            w_next = [w[i]*np.exp(-1*alpha*y*ts_result[i]) for i, y in enumerate(label)]
            z = sum(w_next)
            w = np.array(w_next)/z

            print(v, err, alpha, w)



    def predict(self, data):
        pass

class TreeStump(object):

    def __init__(self):
        self.error = 1
        self.adop_v = 0
        self.result = []

    def fit(self, data, target, weight):
        m = data.shape[0]

        for i in range(m-1):
            v = (data[i] + data[i+1]) / 2
            for j in ["<", ">"]:
                if j == "<":
                    result = [1 if x < v else -1 for x in data]
                else:
                    result = [1 if x > v else -1 for x in data]
                error_list = [0 if x == result[k] else 1*weight[k] for k, x in enumerate(target)]
                error_v = sum(error_list)
                if error_v <self.error:
                    self.adop_v = v
                    self.error = error_v
                    self.result = result
            print(v, self.error, self.adop_v)


if __name__ == "__main__":
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    label = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

    adboost = AdaBoost()
    adboost.fit(data, label)


