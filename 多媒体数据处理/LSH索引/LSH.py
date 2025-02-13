import numpy as np
from numpy import dot, floor, argsort
from numpy.random import random, uniform
import pandas as pd
import numba as nb
import time

# 导入数据集
data = np.loadtxt('corel.txt', usecols=range(1, 33))
# 加载预先算好的前1000个点的真实10近邻
true_idxes = pd.read_csv('true_indices.csv', header=None)
# print(data.shape)

# 计算欧氏距离
@nb.jit(nopython=True)
def calc_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 将向量映射到各桶的索引
def hash_and_fill(inputs, R, b, a):
    # 初始化空的hash_table
    buckets = [{} for _ in range(bucket_num)]
    mapped_idxes = floor((dot(inputs, R) + b) / a)  # 每一行是这个点在所有桶中的哈希值
    for i, hash_keys in enumerate(mapped_idxes):
        # 遍历每个数据点的哈希键。
        for j, hash_key in enumerate(hash_keys):
            # 对于每个数据点，遍历其映射到的每个桶的索引
            # 每个桶是一个字典，其中的所有key对应 该桶的所有索引键值，每个key对应的value是一个list，里面存放映射到该桶、该索引键值的所有点在原数据集的idx
            buckets[j].setdefault(hash_key, []).append(i)
    return buckets
# 数据点的索引添加到对应桶和索引键的列表

def find(q, k, R, b, a, buckets):
    global candi_set
    hash_keys = np.floor((dot(q, R) + b) / a)[0]   # 取[0]转为数组
    # 遍历q点的索引键列表，找各桶中与其索引键值相等的点
    for i, hash_key in enumerate(hash_keys):
        if i == 0:
            candi_set = set(buckets[0][hash_key])
        else:
            candi_set = candi_set.union(buckets[i][hash_key])  # 候选集
    candi_set = list(candi_set)    # 转为list便于遍历
    # 遍历候选集，求出离q最近的k个点并返回
    dist = [calc_dist(data[i], q) for i in candi_set]

    set_idxes = argsort(dist)[1: k + 1]  # set_idxes是近邻点在候选集中的索引，要将其转为在原数据集中的索引
    res = [candi_set[i] for i in set_idxes]
    return res

# tables_num: hash_table的个数
# d: 向量的维数
# a是桶宽。
bucket_num = 15
R = random([32, bucket_num])
a = 0.05
b = uniform(0, a, [1, bucket_num])
buckets = hash_and_fill(data, R, b, a)

# LSH
pred_idx = []
tic = time.time()
for i in range(1000):
    res = find(data[i], 10, R, b , a, buckets)
    pred_idx.append(res)
toc = time.time()
print('耗时：{:.4f}秒'.format(toc - tic))

def count_true_num(pred, true):
    a = [i in true for i in pred]
    return sum(a)

n = 1000
precisions = []
recalls = []
accuracys = []
for i in range(n):
    TP = count_true_num(pred_idx[i], true_idxes.iloc[i,:].to_list())
    FP = 10 - TP
    FN = 10 - TP
    TN = 68040 - 10 - FP
    precisions.append(TP / 10)
    recalls.append(TP / (TP + FN))
    accuracys.append((TP + TN) / (TP + TN + FP + FN))

p, r, acc = np.mean(precisions), np.mean(recalls), np.mean(accuracys)
print("查准率：{:.4f}%\n召回率：{:.4f}%\n准确率：{:.4f}%".format(p*100, r*100, acc*100))
