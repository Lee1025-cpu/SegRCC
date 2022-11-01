# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-07-02 15:32
# 文件名称：k_fold_test
# 开发工具：PyCharm
import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [4, 5]])
y = np.array([1, 2, 3, 4, 5])
kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]