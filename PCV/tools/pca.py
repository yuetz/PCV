# coding=utf-8
import numpy as np


def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        输入： 矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
        返回： 投影 矩阵（ 按照 维 度 的 重要性 排序）、 方差 和 均值
    """

    # get dimensions
    num_data, dim = X.shape

    # center data 数据中心化：减去每一维的均值
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick used 使用紧致技巧
        M = np.dot(X, X.T)  # covariance matrix 协方差矩阵
        e, EV = np.linalg.eigh(M)  # eigenvalues and eigenvectors 特征值和特征向量. eigh求对称阵的（不校验是否是对称阵，只取下三角形）。eig求普通的
        tmp = np.dot(X.T, EV).T  # this is the compact trick 紧致技巧
        V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]  # only makes sense to return the first num_data

    # 返回投影矩阵， 方差， 和均值
    # return the projection matrix, the variance and the mean
    return V, S, mean_X


def center(X):
    """    Center the square matrix X (subtract col and row means). """

    n, m = X.shape
    if n != m:
        raise Exception('Matrix is not square.')

    # colsum = X.sum(axis=0) / n
    # rowsum = X.sum(axis=1) / n
    # totalsum = X.sum() / (n ** 2)

    colsum = np.average(X, axis=0)
    rowsum = np.average(X, axis=1)
    totalsum = np.average(X, axis=(0, 1))   # np.average(X)

    # center
    Y = np.array([[X[i, j] - rowsum[i] - colsum[j] + totalsum for i in range(n)] for j in range(n)])

    return Y