#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    测试logistic regression
"""

from sklearn.datasets import load_iris
from linear_model.logistic_regression import LogisticRegression

data, target = load_iris(True)

train = data[:100]
target = target[:100]

clf = LogisticRegression()
clf.fit(train, target)

result = clf.predict(train)
