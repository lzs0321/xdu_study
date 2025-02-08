import pandas as pd
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as nmi_score

# 加载数据集
df = pd.read_csv('D:/MYpython/MS_homework/BcwCluster/实验数据/wdbc.data', header=None)

# 分离特征和标签
# X = df.iloc[:, 2:].values  # 特征从第3列到第32列
X = df.iloc[:, [29, 24, 9, 22, 4, 25, 2, 5, 8, 28, 7, 27, 12, 14, 15, 23, 26, 30, 3]].values  # 选取指定列
y_true = df.iloc[:, 1].values  # 真实的诊断标签在第2列

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# 聚类
def custom_kmeans(X, n_clusters, max_iter=1000):
    # 初始化聚类中心
    centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
    for _ in range(max_iter):
        # 计算样本到聚类中心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)
        # 分配样本到最近的聚类中心
        labels = np.argmin(distances, axis=-1)
        # 更新聚类中心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        # 判断聚类中心是否变化小于阈值
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# kmeans = KMeans(n_clusters=2, n_init=10,random_state=50)
# y_pred = kmeans.fit_predict(X_scaled)
kmeans_labels, kmeans_centroids = custom_kmeans(X_pca, 2)
dump((kmeans_labels, kmeans_centroids), 'model_2.joblib')

# 评估聚类结果
nmi = nmi_score(y_true, kmeans_labels)
print(f"Normalized Mutual Information (NMI): {nmi}")
