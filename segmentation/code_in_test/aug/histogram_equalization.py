# coding:utf-8

import cv2 as cv

import numpy as np

img = cv.imread('/Now_state/code_in_test/aug/mapel.png')

im_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 将图像转为灰度，减少计算的维度

cv.imshow('im_gray', im_gray)

w = im_gray.shape[0]

h = im_gray.shape[1]

print im_gray

import matplotlib.pyplot as plt

p1 = plt.hist(im_gray.reshape(im_gray.size, 1))

# plt.subplot(121)

plt.show()

# 创建直方图

n = np.zeros((256), dtype=np.float)

p = np.zeros((256), dtype=np.float)

c = np.zeros((256), dtype=np.float)

# 遍历图像的每个像素,得到统计分布直方图

for x in range(0, im_gray.shape[0]):

    for y in range(0, im_gray.shape[1]):

        # print im_gray[x][y]

        n[im_gray[x][y]] += 1

# 归一化

for i in range(0, 256):

    p[i] = n[i] / float(im_gray.size)

# 计算累积直方图

c[0] = p[0]

for i in range(1, 256):

    c[i] = c[i - 1] + p[i]

des = np.zeros((w, h), dtype=np.uint8)

for x in range(0, w):

for y in range(0, h):

des[x][y] = 255 * c[im_gray[x][y]]

print
des

cv.imshow('des', des)

p2 = plt.hist(des.reshape(des.size, 1))

# plt.subplot(121)

plt.show()