# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-04-26 10:52
# 文件名称：gamma_aug
# 开发工具：PyCharm
import cv2
import numpy as np
import math
import SimpleITK as sitk
from matplotlib.pyplot import plot as plt
from pylab import *


def gamma_trans(img, gamma):  # gamma函数处理
    """
    gamma correction
    Args:
        img: H, W, C pic
        gamma: gamma corr

    """
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数

    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def main(exp_flag, nii_flag, nii_in_path, gamma1):
    if exp_flag:
        file_path = "/Now_state/code_in_test/aug/mapel.png"
        img_gray = cv2.imread(file_path, 0)  # 灰度图读取，用于计算gamma值
        img = cv2.imread(file_path)  # 原图读取

        mean = np.mean(img_gray)
        gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
        # gamma_val = 2

        image_gamma_correct = gamma_trans(img[:, :, 1], gamma_val)  # gamma变换
        image_gamma_correct1 = gamma_trans(img_gray, gamma1)  # gamma变换
        # image_gamma_correct2 = gamma_trans(img_gray, 1 / gamma1)  # gamma变换


        fig = plt.figure()
        subplot(131)
        imshow(image_gamma_correct1, cmap='gray')
        title('gamma=' + str(gamma1))
        axis('off')
        subplot(132)
        imshow(img_gray, cmap='gray')
        title('ori')
        axis('off')
        subplot(133)
        imshow(image_gamma_correct, cmap='gray')
        title('gamma=' + str(gamma_val))
        axis('off')
        plt.savefig('gamma_corr.jpg')
        plt.show()

        hist, bins = np.histogram(img_gray, 256, [0, 255])
        # 使用 plt.fill 函数 填充多边形
        plt.fill(hist)
        #  标记x轴的名称
        plt.xlabel('pixel value')
        # 显示直方图
        plt.show()
        plt.savefig('ori hist.png')

        hist, bins = np.histogram(image_gamma_correct1, 256, [0, 255])
        # 使用 plt.fill 函数 填充多边形
        plt.fill(hist)
        #  标记x轴的名称
        plt.xlabel('pixel value')
        # 显示直方图
        plt.show()
        plt.savefig('hist at gamma=' + str(gamma1) + '.png')

        hist, bins = np.histogram(image_gamma_correct, 256, [0, 255])
        # 使用 plt.fill 函数 填充多边形
        plt.fill(hist)
        #  标记x轴的名称
        plt.xlabel('pixel value')
        # 显示直方图
        plt.show()
        plt.savefig('hist at gamma=' + str(gamma_val) + '.png')

        # cv2.namedWindow('image_raw', 0)   # 解决显示不全的问题
        # cv2.imshow('image_raw', img_gray)
        #
        # cv2.namedWindow('image_gamma', 0)
        # cv2.imshow('image_gamma', image_gamma_correct)
        #
        # cv2.waitKey(0)

    if nii_flag:
        img = sitk.GetArrayFromImage(nii_in_path)


if __name__ == '__main__':
    exp_flag = True
    nii_flag = False

    gamma1 = 0.8

    nii_in_path = 'D:/Dataset/1/train/case_00000/imaging.nii.gz'
    main(exp_flag, nii_flag, nii_in_path, gamma1)
