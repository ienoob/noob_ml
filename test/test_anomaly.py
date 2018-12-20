#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.metrics import recall_score, precision_score
from scipy.io import loadmat
from anomaly_detect.statistical_method import mahalanobis_distance, chisquare_test, multi_gaussian, multi_gaussian_dis


data = loadmat("D:\code\git\\noob_ml\dataset\\anomaly_data\\arrhythmia.mat")

big_x = data["X"]
y = data["y"]

print(big_x.shape)
print(y.sum())

# 马氏距离
result = mahalanobis_distance(big_x, top=10)
print(recall_score(y, result), precision_score(y, result))

# 卡方检验
result = chisquare_test(big_x, top=10)
print(recall_score(y, result), precision_score(y, result))

# 多个高斯分布
result = multi_gaussian(big_x, top=10)
print(recall_score(y, result), precision_score(y, result))

# 多元高斯分布
result = multi_gaussian_dis(big_x, top=10)
print(recall_score(y, result), precision_score(y, result))
