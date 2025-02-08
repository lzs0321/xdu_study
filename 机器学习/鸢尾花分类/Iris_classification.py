import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from joblib import dump


# 加载数据集
def load_data(filename):
    return pd.read_csv(filename, header=None)

class KNNModel:
    def __init__(self, training_data, training_labels, k):
        self.training_data = training_data
        self.training_labels = training_labels
        self.k = k

    # 计算欧氏距离
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # 找到K个最近邻居的索引
    def get_neighbors(self, test_instance):
        distances = [self.euclidean_distance(instance, test_instance) for instance in self.training_data]
        sorted_indices = np.argsort(distances)
        return sorted_indices[:self.k]

    # 多数投票预测
    def predict_classification(self, test_instance):
        neighbors = self.get_neighbors(test_instance)
        neighbor_labels = [self.training_labels[i] for i in neighbors]
        most_common = Counter(neighbor_labels).most_common(1)
        return most_common[0][0]


# 主函数
def main():
    # 加载数据集
    train_data = load_data('D:/MYpython/MS_homework/IrisClassification/实验数据/iris_train.data')
    test_data = load_data('D:/MYpython/MS_homework/IrisClassification/实验数据/iris_test.data')
    validation_data = load_data('D:/MYpython/MS_homework/IrisClassification/实验数据/iris_validation.data')

    # 分离特征和标签
    X_train = train_data.iloc[:, :-1].values  # 不包括最后一列，只提取特征
    y_train = train_data.iloc[:, -1].values  # 提取最后一列，标签
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    X_val = validation_data.iloc[:, :-1].values
    y_val = validation_data.iloc[:, -1].values

    # 训练模型
    # training_data, training_labels = X_train, y_train
    # 选择K值
    k = 3  # 可以在这里使用交叉验证来选择最佳的K值

    # 保存模型
    model = KNNModel(training_data=X_train, training_labels=y_train, k=k)
    dump(model, 'model_1.joblib')

    # 在验证集上评估模型
    correct_predictions = 0
    for x_val, y_val_label in zip(X_val, y_val):
        prediction = model.predict_classification(x_val)
        if prediction == y_val_label:
            correct_predictions += 1
    # 计算准确率
    validation_accuracy = correct_predictions / len(y_val)
    print(f"Validation Accuracy: {validation_accuracy}")

    # 在测试集上评估模型
    correct_predictions = 0
    for x_test, y_test_label in zip(X_test, y_test):
        prediction = model.predict_classification(x_test)
        if prediction == y_test_label:
            correct_predictions += 1
    test_accuracy = correct_predictions / len(y_test)
    print(f"Test Accuracy: {test_accuracy}")




if __name__ == "__main__":
    main()