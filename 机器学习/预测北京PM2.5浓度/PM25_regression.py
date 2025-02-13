import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from joblib import dump

# 读取数据文件
train_path = 'D:/MYpython/MS_homework/PM2_5Regression/实验数据/PRSA_train.data1.csv'
test_path = 'D:/MYpython/MS_homework/PM2_5Regression/实验数据/PRSA_test.data1.csv'
validation_path = 'D:/MYpython/MS_homework/PM2_5Regression/实验数据/PRSA_validation.data1.csv'

# 读取训练集、测试集和验证集的数据
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
validation = pd.read_csv(validation_path)

# 预处理数据函数
def preprocess_data(data):
    data = data.copy()  # 创建数据副本以避免警告
    data.drop('No', axis=1, inplace=True)  # 删除No列
    # 删除缺失值
    data.dropna(subset=['pm2.5'], inplace=True)
    # data = data[data['pm2.5'] <= 400]
    # data['pm2.5'].fillna(0, inplace=True)
    return data

# 预处理训练集、测试集和验证集
train = preprocess_data(train)
test = preprocess_data(test)
validation = preprocess_data(validation)

# 对分类特征进行整数编码
encoder = LabelEncoder()
train['cbwd'] = encoder.fit_transform(train['cbwd'])
test['cbwd'] = encoder.transform(test['cbwd'])
validation['cbwd'] = encoder.transform(validation['cbwd'])

# 分离特征和目标变量
feature_columns = ['year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']
train_X = train[feature_columns]
train_y = train['pm2.5']
test_X = test[feature_columns]
test_y = test['pm2.5']
validation_X = validation[feature_columns]
validation_y = validation['pm2.5']


# 训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_X, train_y)

# 保存模型
dump(rf_model, 'model_3.joblib')
# rf_model_loaded = load('model_3.joblib')

# 在测试集和验证集上进行预测
test_predictions = rf_model.predict(test_X)
validation_predictions = rf_model.predict(validation_X)

# 计算测试集的均方根误差（RMSE）
rmse = sqrt(mean_squared_error(test_y, test_predictions))
print('测试集的均方根误差（RMSE）: %.3f' % rmse)

# 计算验证集的均方根误差（RMSE）
rmse_V = sqrt(mean_squared_error(validation_y, validation_predictions))
print('验证集的均方根误差（RMSE）: %.3f' % rmse_V)
# 可视化预测结果
num_points_to_visualize = 600

# 确保我们不会尝试可视化的数据点超过我们实际拥有的数量
num_points_to_visualize = min(num_points_to_visualize, len(test_y), len(validation_y))

# 选择数据点
test_actual = test_y[:num_points_to_visualize]
test_predicted = test_predictions[:num_points_to_visualize]

validation_actual = validation_y[:num_points_to_visualize]
validation_predicted = validation_predictions[:num_points_to_visualize]

index = range(num_points_to_visualize)

# 创建折线图
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# 测试集的实际值和预测值
axs[0].plot(test_actual.index, test_actual, label='Test Actual', marker='.', color='blue')
axs[0].plot(test_actual.index, test_predicted, label='Test Predicted', marker='.', color='red')
axs[0].set_title('Test Set Predictions vs Actual')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('PM2.5 Concentration')
axs[0].legend()

# 验证集的实际值和预测值
axs[1].plot(validation_actual.index, validation_actual, label='Validation Actual', marker='x', color='green')
axs[1].plot(validation_actual.index, validation_predicted, label='Validation Predicted', marker='x', color='orange')
axs[1].set_title('Validation Set Predictions vs Actual')
axs[1].set_xlabel('Index')
axs[1].legend()

# 显示图
plt.tight_layout()
plt.show()