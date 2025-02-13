import numpy as np

def get_asc_file_dimensions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    ncols = len(lines[0].split())
    return ncols  # 获得列数

def custom_pca(dataset, num_components):
    # 均值中心化
    centered_data = dataset - np.mean(dataset, axis=0)

    # 计算协方差矩阵
    # C = (1 / (n-1)) * X_centered^T * X_centered
    # 利用公式来计算协方差矩阵 C，其中 X_centered^T 表示 X_centered 的转置。
    # 获取维度(列数)
    ncols = get_asc_file_dimensions(file_path)
    covariance_matrix =(1 / (ncols-1)) * np.dot(centered_data.T, centered_data)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 对特征值和特征向量进行排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前num_components个主成分
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # 将数据投影到选定的主成分上
    projected_data = np.dot(centered_data, selected_eigenvectors)

    # 计算PCA之前的数据方差
    variance_before_pca = np.var(dataset, axis=0)

    # 计算PCA之后的数据方差
    variance_after_pca = np.var(projected_data, axis=0)

    return variance_before_pca, projected_data, variance_after_pca


# 读取数据集
dataset = np.loadtxt('ColorHistogram.asc')
file_path = "D:\MYpython\PCA_deal\ColorHistogram.asc"
# 进行主成分分析
variance_before_pca, projected_data, variance_after_pca= custom_pca(dataset, num_components=5)

# 输出结果
print("PCA之前数据方差:\n", variance_before_pca)
print("PCA之后的数据（降至5维）:")
print(projected_data)
print("PCA之后数据方差:\n", variance_after_pca)