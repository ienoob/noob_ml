#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from ann.multiply_perceptron import MultiplyPerceptron


data, target = load_iris(True)

train = data[:100]
target = target[:100]

print(train.shape)

mp = MultiplyPerceptron()
mp.fit(train, target)