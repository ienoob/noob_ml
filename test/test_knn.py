#!/usr/bin/env python
# -*- coding:utf-8 -*-
import unittest
from nearest_neighbor.k_nearest_neighbor import KNN, KDTree


class TestKNN(unittest.TestCase):
    def test_predict(self):
        train_data = [[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]
        target_data = [1, 1, 0, 0]

        knn = KNN()
        knn.fit(train_data, target_data)

        test_data = [[1.0, 1.2], [0, 0.2]]

        predict_dat = knn.predict(test_data)

        self.assertEqual(predict_dat, [1, 0])

        # 出bug了
        knn.fit_usekd(train_data, target_data)
        predict_dat = knn.predict(test_data, p_type="kdtree")
        # print(predict_dat)
        #
        # self.assertEqual(predict_dat, [1, 0])

    def test_kdtree(self):
        train_data = [[2, 3, -1], [5, 4, -1], [9, 6, 1], [4, 7, -1], [8, 1, 1], [7, 2, 1]]
        kdt = KDTree()
        kdt.build(train_data, 2)

        self.assertEqual(kdt.root.split_feature, 0)
        self.assertEqual(kdt.root.x, [7, 2, 1])
        self.assertEqual(kdt.root.parent, None)

        self.assertEqual(kdt.root.left.split_feature, 1)
        self.assertEqual(kdt.root.left.parent, kdt.root)
        self.assertEqual(kdt.root.right.split_feature, 1)
        self.assertEqual(kdt.root.right.parent, kdt.root)

        self.assertEqual(kdt.root.left.x, [5, 4, -1])
        self.assertEqual(kdt.root.right.x, [9, 6, 1])

        self.assertEqual(kdt.root.left.left.split_feature, None)
        self.assertEqual(kdt.root.left.right.split_feature, None)

        self.assertEqual(kdt.root.left.left.x, [2, 3, -1])
        self.assertEqual(kdt.root.left.right.x, [4, 7, -1])

        self.assertEqual(kdt.root.right.left.split_feature, None)
        self.assertEqual(kdt.root.right.right, None)

        self.assertEqual(kdt.root.right.left.x, [8, 1, 1])

        self.assertEqual(kdt.search_knn([4, 4]), [(1.0, -1)])


if __name__ == "__main__":
    unittest.main()
