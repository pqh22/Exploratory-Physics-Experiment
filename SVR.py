import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
import math

# 读取数据
X = np.load('大物实验/theta.npy')
y = np.load('大物实验/v.npy')
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建SVR模型，包括RBF、线性和多项式核函数
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100)
svr_poly = SVR(kernel="poly", C=100, degree=3)

# 使用网格搜索来寻找最佳超参数
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': [0.1, 1],
}
grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_svr = grid_search.best_estimator_

# 用于可视化的设置
lw = 2
svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ["RBF", "Linear", "Polynomial"]
model_color = ["m", "c", "g"]

# 创建子图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

# 训练模型并绘制结果
for ix, svr in enumerate(svrs):
    y_pred = svr.fit(X_train, y_train).predict(X)
    axes[ix].plot(X, y_pred, color=model_color[ix], lw=lw, label="{} model".format(kernel_label[ix]))
    axes[ix].scatter(X_train, y_train, facecolor="none", edgecolor="k", s=50, label="training data")
    axes[ix].scatter(X_test, y_test, facecolor="none", edgecolor="r", s=50, label="testing data")
    axes[ix].legend()
    axes[ix].set_title("SVR with {} kernel".format(kernel_label[ix]))
    axes[ix].text(0.02, 0.75, "MSE: {:.2f}".format(mean_squared_error(y, y_pred)), transform=axes[ix].transAxes)
    axes[ix].text(0.02, 0.65, "R2: {:.2f}".format(r2_score(y, y_pred)), transform=axes[ix].transAxes)

# fig.text(0.5, 0.04, "theta/rad", ha="center", va="center")
# fig.text(0.06, 0.5, "v_0/m·s^-1", ha="center", va="center", rotation="vertical")
# fig.suptitle("Support Vector Regression", fontsize=14)
# plt.show()

# 输出每组超参数的模型的平均得分与标准差
cv_results = grid_search.cv_results_
for mean_score, std_score, params in zip(cv_results['mean_test_score'], cv_results['std_test_score'], cv_results['params']):
    print("超参数: ", params)
    print("平均得分: {:.2f}".format(mean_score))
    print("标准差: {:.2f}".format(std_score))
    print()


# 打印最佳超参数和评估指标
print("最佳超参数:", grid_search.best_params_)
y_pred_train = best_svr.predict(X_train)
y_pred_test = best_svr.predict(X_test)
print("训练集MSE:", mean_squared_error(y_train, y_pred_train))
print("训练集R2:", r2_score(y_train, y_pred_train))
print("测试集MSE:", mean_squared_error(y_test, y_pred_test))
print("测试集R2:", r2_score(y_test, y_pred_test))





