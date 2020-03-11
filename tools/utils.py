#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: liucheng
@contact: liucheng@memect.co
@file: utils.py
@time: 2/29/20 9:43 PM
@desc:
"""
import numpy as np


def sigmoid(z):
    """
    sigmoid 激活函数
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(z))


def sigmoid_derivative(a):
    """
    sigmoid 求导
    注意：这里传入的a值，而不是z值
    :param a:
    :return:
    """
    return np.multiply(a, (1 - a))


def init_thetas(hidden_num, unit_num, input_size, class_num, epsilon):
    """
    初始化权值矩阵
    :param hidden_num: 隐藏层数目
    :param unit_num: 每个隐藏层的神经元数目
    :param input_size: 输出层规模（维度）
    :param class_num: 分类数目
    :param epsilon: 权值上限
    :return: 权值矩阵序列
    """
    # 隐藏层每层神经元个数
    hiddens = [unit_num for i in range(hidden_num)]
    # 输出层 + 隐藏层 + 输出层，所有层的神经元个数
    units = [input_size] + hiddens + [class_num]
    # 输出值
    thetas = []

    for idx, unit in enumerate(units):
        # 权值个数 = 层数 - 1
        if idx == len(units) - 1:
            break
        # 获取下一层的神经元个数
        next_unit = units[idx+1]

        # *** 初始化权值 ***
        # 生成 当前层 -> 下一层 的初始权值矩阵，值的大小属于：[0, 1]
        rand_weight = np.random.rand(next_unit, unit + 1)
        # 将权值大小归一化到：[-epsilon, epsilon] 范围内
        theta = rand_weight * 2 * epsilon - epsilon

        thetas.append(theta)

    return thetas


def compute_cost(thetas, y, the_lambda, x=None, a=None):
    """
    计算代价
    :param thetas: 权值矩阵序列
    :param y: 标签集
    :param the_lambda: 正则化参数
    :param x: 样本
    :param a: 各层激活值（是列表）
    :return: 预测代价
    """
    m = y.shape[0]
    if a is None:
        # 通过 "前向传播" 计算各层激活值
        a = forward_propagation(thetas, x)

    # 注意，计算代价的时候，只需要关注整个网络的预测和标注之间的差异即可，因此只需要看 a[-1]
    # 另外一个注意点是：标注 y 已经被向量化了，有且仅有一位是1，其他都是0
    error = -np.sum(np.multiply(y.T, np.log(a[-1])) + np.multiply((1 - y).T, np.log(1 - a[-1])))

    # 正则化项，但不包括偏置项，Θ 的下标 i 是下一层的神经元编号，下标 j 是当前层的节点编号。所以偏置项在第二维的第0个位置
    reg = -np.sum([np.sum(theta[:, 1:]) for theta in thetas])

    return (1.0 / m) * error + (1.0 / (2 * m)) * the_lambda * reg


def adjust_labels(y):
    """
    标签向量化
    :param y: 标签集
    :return: 向量化后的标签
    """
    if y.shape[1] == 1:
        classes = set(np.ravel(y))
        class_num = len(classes)
        min_class = min(classes)

        if class_num > 2:
            y_adjusted = np.zeros((y.shape[0], class_num), np.float64)
            for row, label in enumerate(y):
                if label != min_class:
                    y_adjusted[row, label - min_class] = 1
        else:
            y_adjusted = np.zeros((y.shape[0], class_num), np.float64)
            for row, label in enumerate(y):
                if label != min_class:
                    y_adjusted[row, 0] = 1.0

        return y_adjusted

    return y


def unroll(matrixes):
    """
    参数展开
    :param matrixes: 矩阵
    :return: 向量
    """
    vec = []
    for matrix in matrixes:
        vector = matrix.reshape(1, -1)[0]
        vec = np.concatenate((vec, vector))
    return vec


def roll(vector, shapes):
    """
    参数恢复
    :param vector: 向量
    :param shapes: 维度列表
    :return: 恢复的矩阵序列
    """
    matrixes = []
    begin = 0
    for shape in shapes:
        end = begin + shape[0] * shape[1]
        matrix = vector[begin:end].reshape(shape)
        begin = end
        matrixes.append(matrix)
    return matrixes


def forward_propagation(thetas, x):
    """
    前向反馈过程
    :param thetas: 权值矩阵
    :param x: 输入样本
    :return: 各层激活向量
    """
    # 获取网络层数
    layer_num = len(thetas) + 1
    # 定义激活向量序列
    a = [None] * layer_num

    # 前向传播计算各层输出
    for i in range(layer_num):
        # 第一层的值，即为输入样本的转置
        if i == 0:
            a[i] = x.T
        else:
            # 计算线性求和值，这里z是一个列数为m（样本个数），行数为s（当前层神经元个数）的矩阵
            z = thetas[i - 1] * a[i - 1]
            # a[i]为行为s，列数为m的矩阵
            a[i] = sigmoid(z)

        # 除输出层外，需要添加偏置项
        if i != layer_num - 1:
            # 生成偏置项，维度为 (1, m)
            bias = np.ones((1, a[i].shape[1]))
            # 添加偏置项
            a[i] = np.concatenate((bias, a[i]))

    return a


def backward_propagation(thetas, a, y, the_lambda):
    """
    反向传播过程
    :param thetas: 各层权值
    :param a: 各层激活值
    :param y: 标签
    :param the_lambda: 正则化参数
    :return: 权值梯度
    """
    # 获取样本个数
    m = y.shape[0]
    # 获取网络层数
    layer_num = len(thetas) + 1
    # 定长数组
    d = [None] * layer_num
    # 初始化误差为0
    delta = [np.zeros(theta.shape) for theta in thetas]

    for i in range(layer_num)[::1]:    # 反向遍历每一层
        # 输入层不计算误差
        if i == 0:
            break
        if i == layer_num - 1:
            # 输出层误差
            d[i] = a[i] - y.T
        else:
            # 忽略偏置
            d[i] = np.multiply((thetas[i][:, 1:].T * d[i + 1]), sigmoid_derivative(a[i][1:, :]))

    for i in range(layer_num - 1):
        delta[i] = d[i + 1] * a[i].T

    weight_gradient = [np.zeros(theta.shape) for theta in thetas]
    for i in range(layer_num):
        theta = thetas[i]
        # 偏置更新增量
        weight_gradient[i][:, 0] = (1.0 / m) * (delta[i][0:, 0].reshape(1, -1))
        # 权值更新增量
        weight_gradient[i][:, 1:] = (1.0 / m) * (delta[i][0:, 1:] + the_lambda * theta[:, 1:])

    return weight_gradient
