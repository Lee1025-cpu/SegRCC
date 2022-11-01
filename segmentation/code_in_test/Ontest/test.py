# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2020-12-21 17:06
# 文件名称：test
# 开发工具：PyCharm
import SimpleITK as sitk
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn


img = mpimg.imread('C:/Users/HP/Desktop/Snipaste_2020-12-11_10-22-03.png')
plt.imshow(img)
plt.show()

plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()
# x = sitk.ReadImage('D:/Dataset/1/train/case_00000/imaging.nii.gz')
# x1 = nb.load('D:/Dataset/1/train/case_00000/imaging.nii.gz')
# affine = x1.affine
# data = x1.get_fdata()
# print()
a = np.random.randint(0, 10, size=[5, 3, 3])
print(a)
# for i in range(a.shape[0]):
#     a[i, :, :] = np.fliplr(a[i, :, :])
b = np.fliplr(a)
print(b)
c = np.flipud(a)
print(c)


# 卷积核赋值
numpy_data= np.random.randn(6, 1, 3, 3)

conv = nn.Conv2d(1, 6, 3, 1, 1, bias=False)
with torch.no_grad():
    conv.weight = nn.Parameter(torch.from_numpy(numpy_data).float())
    # or
    conv.weight.copy_(torch.from_numpy(numpy_data).float())
