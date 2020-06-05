#!/usr/bin/env python
# -*- coding:utf-8 -*-

from decision_trees.id3 import ID3
features = ["年龄", "有工作", "有自己的房子", "信贷情况"]
X = [["青年", "否", "否", "一般"],
     ["青年", "否", "否", "好"],
     ["青年", "是", "否", "好"],
     ["青年", "是", "是", "一般"],
     ["青年", "否", "否", "一般"],
     ["中年", "否", "否", "一般"],
     ["中年", "否", "否", "好"],
     ["中年", "是", "是", "好"],
     ["中年", "否", "是", "非常好"],
     ["中年", "否", "是", "非常好"],
     ["老年", "否", "是", "非常好"],
     ["老年", "否", "是", "好"],
     ["老年", "是", "否", "好"],
     ["老年", "是", "否", "非常好"],
     ["老年", "否", "否", "一般"]]
Y = [
     "否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"
]

id3 = ID3()
id3.fit(X, Y)

test_input = ["中年", "否", "是", "非常好"]
print(id3.predict(test_input))
