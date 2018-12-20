#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    统计学方法
    -- ksigma
    -- grubbs test
    -- 卡方统计
    -- 马氏距离
    -- 多个高斯分布和多元高斯分布
"""
from functools import reduce
import numpy as np


def ksigma(x, k=3):

    miu = np.mean(x)
    sigma = np.std(x, ddof=1)

    anomaly_data = []
    for v in x:
        if np.abs(v-miu)/sigma > k:
            anomaly_data.append(1)
        else:
            anomaly_data.append(0)
    return anomaly_data


# 缺点：需要去查询studenct分布, t 在 alpha/2N显著程度，N-2自由度
def grubbs_test(x, t, alpha=0.05):
    big_n = len(x)
    miu = np.mean(x)
    sigma = np.std(x, ddof=1)

    anomaly_data = []
    grubbs_value = big_n*np.sqrt(t**2/(big_n-2+t**2))/np.sqrt(big_n)
    for v in x:
        if np.abs(v-miu)/sigma >= grubbs_value:
            anomaly_data.append(1)
        else:
            anomaly_data.append(0)
    return anomaly_data


# 马氏距离
def mahalanobis_distance(x, threshold=0, top=1):
    vector_miu = np.mean(x, axis=0)

    # 去中心化
    # no_center = x - vector_miu
    # cover_sigma = np.dot(no_center.T, no_center)/(x.shape[0]-1)

    # 求协方差
    cover_sigma = np.cov(x.T, ddof=1)
    if np.linalg.det(cover_sigma) == 0:
        cover_sigma_inv = np.linalg.pinv(cover_sigma)
    else:
        cover_sigma_inv = np.linalg.inv(cover_sigma)

    anomaly_data = []
    for v in x:
        minus_miu = v - vector_miu
        # 马氏距离
        maha_dist = np.sqrt(np.dot(np.dot(minus_miu.T, cover_sigma_inv), minus_miu))
        if threshold > 0:
            if maha_dist > threshold:
                anomaly_data.append(1)
            else:
                anomaly_data.append(0)
        else:
            anomaly_data.append(maha_dist)
    if top > 0:
        topk_value = sorted(anomaly_data, reverse=True)[top]
        anomaly_data = [0 if v < topk_value else 1 for v in anomaly_data]
    return anomaly_data


# 卡方检验
def chisquare_test(x, threshold=None, top=0):
    vector_miu = np.mean(x, axis=0)
    vector_miu += 1
    x += 1

    anomaly_data = []
    for v in x:
        minus_miu = v - vector_miu
        chi_value = np.dot(minus_miu.T, minus_miu/vector_miu)
        # chi_value = np.sum(np.power(v-vector_miu, 2)/vector_miu)
        if threshold is None:
            anomaly_data.append(chi_value)
        else:
            if chi_value > threshold:
                anomaly_data.append(1)
            else:
                anomaly_data.append(0)
    if top > 0:
        topk_value = sorted(anomaly_data, reverse=True)[top]
        anomaly_data = [0 if v < topk_value else 1 for v in anomaly_data]
    return anomaly_data


# 多个高斯分布线性组和
def multi_gaussian(x, threshold=0, top=1):
    vector_miu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)

    anomaly_data = []
    for v in x:
        d = np.exp(-1*np.power(v-vector_miu, 2)/2/np.power(sigma, 2))/np.sqrt(2*np.pi)/sigma
        multi_d = reduce(lambda x,y: x*y, d)
        if threshold > 0:
            if multi_d < threshold:
                anomaly_data.append(1)
            else:
                anomaly_data.append(0)
        else:
            anomaly_data.append(multi_d)
    if top > 0:
        topk_value = sorted(anomaly_data)[top]
        anomaly_data = [1 if v < topk_value else 0 for v in anomaly_data]
    return anomaly_data


# 多元高斯分布
def multi_gaussian_dis(x, threshold=0, top=0):
    big_n = x.shape[0]
    vector_miu = np.mean(x, axis=0)
    cover_sigma = np.cov(x.T, ddof=1)
    if np.linalg.det(cover_sigma) == 0:
        cover_sigma_inv = np.linalg.pinv(cover_sigma)
    else:
        cover_sigma_inv = np.linalg.inv(cover_sigma)
    cover_sigma_det = np.linalg.det(cover_sigma)
    if cover_sigma_det == 0:
        cover_sigma_det = 1

    anomaly_data = []
    for v in x:
        minus_miu = v - vector_miu
        part1 = np.dot(np.dot(minus_miu.T, cover_sigma_inv), minus_miu)
        multi_value = np.exp(-1*part1/2)/np.power(2*np.pi, big_n/2)/np.sqrt(cover_sigma_det)
        if threshold > 0:
            if multi_value < threshold:
                anomaly_data.append(1)
            else:
                anomaly_data.append(0)
        else:
            anomaly_data.append(multi_value)
    if top > 0:
        topk_value = sorted(anomaly_data)[top]
        anomaly_data = [1 if v < topk_value else 0 for v in anomaly_data]
    return anomaly_data
