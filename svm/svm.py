#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    svm 可以转化为二次规划问题，因此我们使用cvxopt， 优化计算包
    hard margin
"""
import numpy as np
from cvxopt import matrix, solvers

A = matrix([[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0]])
b = matrix([1.0, -2.0, 0.0, 4.0])
c = matrix([2.0, 1.0])

sol = solvers.lp(c,A,b)

print(sol['x'])
print(np.dot(sol['x'].T, c))
print(sol['primal objective'])