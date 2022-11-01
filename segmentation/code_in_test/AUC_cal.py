# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-05-30 11:06
# 文件名称：code_in_test
# 开发工具：PyCharm
import numpy as np
from sklearn import metrics

# TODO: AUC code
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
print(metrics.auc(fpr, tpr))

