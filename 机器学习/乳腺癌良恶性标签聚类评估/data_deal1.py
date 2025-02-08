import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 假设df是你的DataFrame，且第二列是类别标签，且类别标签是'B'和'M'
df = pd.read_csv('D:/MYpython/MS_homework/BcwCluster/实验数据/wdbc.data', header=None)

# 分离特征和标签
X = df.iloc[:, 2:].values  # 特征从第3列到第32列
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_true = df.iloc[:, 1].astype('category').cat.codes  # 将标签转换为数值编码，'B'为0，'M'为1

# 初始化一个列表来存储每个特征的分离度
separation_degrees = []
color_map = {0: 'blue', 1: 'red'}
# 计算每个特征的分离度
for j in range(X_scaled.shape[1]):
    class_0_values = X_scaled[y_true == 0, j]
    class_1_values = X_scaled[y_true == 1, j]
    mean_class_0 = np.mean(class_0_values)
    mean_class_1 = np.mean(class_1_values)
    # 欧氏距离作为分离度的度量
    separation_degree = np.abs(mean_class_0 - mean_class_1)
    separation_degrees.append((j, separation_degree))

# 根据分离度对特征进行排序
sorted_features = sorted(separation_degrees, key=lambda x: x[1], reverse=True)

# 输出相似度最低的前30个特征的列数（索引+1，因为用户希望得到列数）
for i, feature in enumerate(sorted_features[:30]):
    print(f"Feature {feature[0] + 1} has a separation degree of {feature[1]}")

# 可视化分离度最高的前30个特征的散点图
fig, axs = plt.subplots(6, 5, figsize=(15, 6))  # 6行5列的子图布局
axs = axs.ravel()

for i, (feature_index, _) in enumerate(sorted_features[:30]):
    axs[i].scatter(X[:, feature_index], y_true, c=[color_map[y] for y in y_true], alpha=0.5)
    axs[i].set_title(f"Feature {feature_index + 1}")
    axs[i].set_xlabel("Value")
    axs[i].set_ylabel("Class")

plt.tight_layout()
plt.show()