# coding=utf-8
import numpy as np
import pickle

"""
This generates the 2D point data sets used in chapter 8.
(both training and test)
"""


def generate(n, normal_path, ring_path):
    # two normal distributions
    class_1 = 0.6 * np.random.randn(n, 2)  # 均值为[0, 0], 标准差为0.6的正态分布
    class_2 = 1.2 * np.random.randn(n, 2) + np.array([5, 1])  # 均值为[5, 1], 标准差为1.2的正态分布
    labels = np.hstack((np.ones(n), -np.ones(n)))

    # save with Pickle
    with open(normal_path, 'w') as f:
        pickle.dump(class_1, f)
        pickle.dump(class_2, f)
        pickle.dump(labels, f)

    # normal distribution and ring around it 正态分布并使数据成环绕分布
    class_1 = 0.6 * np.random.randn(n, 2)

    r = 0.8 * np.random.randn(n, 1) + 5
    angle = 2 * np.pi * np.random.randn(n, 1)
    class_2 = np.hstack((r * np.cos(angle), r * np.sin(angle)))
    labels = np.hstack((np.ones(n), -np.ones(n)))

    # save with Pickle
    with open(ring_path, 'w') as f:
        pickle.dump(class_1, f)
        pickle.dump(class_2, f)
        pickle.dump(labels, f)


if __name__ == '__main__':
    # Training data : create sample data of 2D points
    generate(n=200, normal_path='../data/points_normal.pkl', ring_path='../data/points_ring.pkl')
    # Test data  : create sample data of 2D points
    generate(n=200, normal_path='../data/points_normal_test.pkl', ring_path='../data/points_ring_test.pkl')

