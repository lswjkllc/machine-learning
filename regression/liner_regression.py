#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: liucheng
@contact: liucheng@memect.co
@file: liner_regression.py
@time: 2/29/20 9:43 PM
@desc:
"""
import numpy as np


"""
predict(X) = XW
X: m,n
W: m,1
h(X): m,1
"""
def predict(data, weight):
    """
    计算单个数据的预测值
    :param data: 数据
    :param weight: 权重
    :return:
    """
    return np.dot(data, weight)


"""
loss(X, Y, W) = np.sum(np.dot(diff.T, diff) / 2 * m)
diff = predict(data, weight) - label
diff: m,1
diff.T: 1,m
np.dot(diff.T, diff): 1,1
"""
def loss(data, label, weight):
    """
    损失函数
    :param data: 所有数据
    :param label: 所有真实值
    :param weight: 权重
    :return:
    """
    m, n = data.shape
    diff = predict(data, weight) - label
    return np.sum(np.dot(diff.T, diff) / 2 * m)


"""
batch_grad_desc(a, max_loop, epsilon, X, Y) = ???
X: m,n
Y: m,1
W: n,1
diff: m,1
slope = np.dot(data.T, diff)
W = W - slope * alpha / m
slope: n,1
"""
def batch_grad_desc(alpha, max_loop, epsilon, data, label):
    """
    批量梯度下降（每次计算都是用全部数据）
    :param alpha:
    :param max_loop:
    :param epsilon:
    :param data:
    :param label:
    :return:
    """
    # m为样本量，n为特征数量
    m, n = data.shape
    # 初始化权重
    weight = np.zeros((n, 1))
    # 记录迭代次数
    count = 0
    # 初始化损失函数的值为
    floss = np.inf
    # 记录每次损失函数的值
    losses = [floss]
    # 用于判断当前损失函数的值和上一次的值是否满足小于epsilon
    lt_epsilon = False

    while count <= max_loop:
        if lt_epsilon:
            break
        count += 1

        diff = predict(data, weight) - label
        slope = np.dot(data.T, diff)
        weight = weight - slope * alpha * 1.0 / m

        the_loss = loss(data, label, weight)
        losses.append(the_loss)
        if losses[-2] - losses[-1] < epsilon:
            lt_epsilon = True

    return weight


if __name__ == '__main__':
    x = np.array([
        [1, 2, 3],
        [2, 5, 6],
        [3, 4, 7],
        [4, 3, 8],
        [5, 9, 6]
    ])
    y = np.array([[1, 2, 3, 4, 5]]).T
    w = batch_grad_desc(0.001, 200, 0.000001, x, y)
    pass
