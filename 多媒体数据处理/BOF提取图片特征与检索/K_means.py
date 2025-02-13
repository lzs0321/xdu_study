import random
import numpy as np

# 计算新的聚类中心
def calcNewCentroids(cls_pos):
    Centroids_new = np.zeros((1, 128))
    for Centroid in cls_pos.keys():
        # 初始化新聚类中心坐标为128维0向量
        pos_new = np.zeros((1, 128))
        for pos in cls_pos[Centroid]:
            pos_new += pos    # 将该类中所有样本点坐标求和
        # 以各类中所有点的坐标的均值作为新的聚类中心点坐标
        pos_new /= len(cls_pos[Centroid])
        # 将新中心坐标添加到列表中
        Centroids_new = np.row_stack((Centroids_new, pos_new))
    return Centroids_new[1:,:]  # 不要第一行


def kmeans(X, k):
    n = X.shape[0]   # 数据点总数
    epoch = 0        # 迭代次数
    idxs = [random.randint(0,k) for _ in range(k)]
    Centroids = X[idxs] # 构造初始的中心点矩阵（1000x128）
    while(1):
        epoch += 1
        cls_pos = {}
        # 对于每个点，计算它到各中心的距离，得到它所属的类
        cls, _ = vq(X, Centroids)
        for i in range(n):
            cls_pos.setdefault(cls[i],[]).append(X[i])         # cls_pos：以类为key，其value为一个数组，保存属于该类的点坐标
        for j in range(k):
            if j not in cls_pos.keys():
                cls_pos.setdefault(j,[]).append(Centroids[j])  # 若某类没有点被分配进来，则该类只有类中心点一个点
        # 所有点均已分到相应的类中。下求新的聚类中心坐标
        Centroids_new = calcNewCentroids(cls_pos)
        # 当聚类中心不再变化时，停止迭代
        if (Centroids_new == Centroids).all():
            break
        # 更新聚类中心为新的聚类中心
        Centroids = Centroids_new
    # 返回聚类中心坐标及迭代次数
    return epoch, Centroids

# k为聚类数 调包计算时间：100/15.2s  500/26.2s  1000/32s  5000/43.5s  10000/1m5s  15000/1m36s
k = 1000
# words_num是视觉词汇数量（聚类中心数）
words_num = k
# 每隔10个descriptor采样一次
sampling_rate = 10
# voc是聚类中心数组，k行128列
# voc, _ = cluster.kmeans(descriptors[::sampling_rate,:],k,1)
epoch, voc = kmeans(descriptors[::sampling_rate,:], k)  # 1000 1min11s