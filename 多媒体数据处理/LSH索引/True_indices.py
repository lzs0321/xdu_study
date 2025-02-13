import numpy as np
from sklearn.neighbors import NearestNeighbors
# 加载数据集
data = np.loadtxt('D:/corel', usecols=range(1, 33))
# 创建模型
model = NearestNeighbors(n_neighbors=11)  # n_neighbors设置为11，因为最近的邻居是样本点本身
model.fit(data)
# 查询前1000个样本点的最近邻
indices = model.kneighbors(data[:1000], return_distance=False)

# 由于最近的邻居是样本点本身，所以我们排除第一列（自身的索引）
true_indices = indices[:, 1:]
np.savetxt('true_indices.csv', true_indices, fmt='%d', delimiter=',')