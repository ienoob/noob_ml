#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from ann.multiply_perceptron import MultiplyPerceptron
from sklearn.neural_network import MLPClassifier


data, target = load_iris(True)

train = data[:100]
target = target[:100]

print(train.shape)

mp = MultiplyPerceptron()
mp.fit(train, target)

result = mp.predict(train)
# print result

# mp2 = MLPClassifier(max_iter=1000)
# mp2.fit(train, target)

