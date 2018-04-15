#!/usr/bin/env python
# -*- coding:utf-8 -*-
from nearest_neighbor.k_nearest_neighbor import KNN

train_data = [[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]
target_data = [1, 1, 0, 0]

knn = KNN()
knn.fit(train_data, target_data)

test_data = [[1.0, 1.2], [0, 0.2]]

predict_dat = knn.predict(test_data)
print(predict_dat)