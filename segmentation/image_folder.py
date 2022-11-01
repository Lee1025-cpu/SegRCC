# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2020-12-25 10:34
# 文件名称：image_folder
# 开发工具：PyCharm
import os
from torch.utils import data
import numpy as np
import cv2
import math
from skimage.util import random_noise
from scipy.ndimage import zoom
import SimpleITK as sitk
import torch


def get_loader_pre(mode, config):
    dataset = Imagefolder_pre(mode, config)
    data_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=0, pin_memory=False, shuffle=True)

    return data_loader


def get_loader(mode, config):
    dataset = Imagefolder(mode, config)
    data_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=0, pin_memory=False, shuffle=True)

    return data_loader


class Imagefolder(data.Dataset):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

        if self.mode == 'run':
            self.in_path = config.run_path

            self.case_names = os.listdir(self.in_path)

        else:
            self.in_path = config.img_aug_saving if config.phase == 1 else config.img_aug_saving_phase3
            self.case_names = []

            if self.mode == 'train':
                for i in range(config.train_idx.shape[0]):
                    self.case_names.append(config.k_fold_lists[config.train_idx[i]])

            elif self.mode == 'valid':
                for i in range(config.valid_idx.shape[0]):
                    self.case_names.append(config.k_fold_lists[config.valid_idx[i]])

        self.img_aug_saving = config.img_aug_saving if config.phase == 1 else config.img_aug_saving_phase3
        print('For {} case counts for {} in {}'.format(self.mode, len(self.case_names), self.in_path))

    def spacing(self):
        resampler = sitk.ResampleImageFilter()
        target_spacing = [3.2, 1.6, 1.6]

        img = sitk.ReadImage(self.in_path + self.case_name)
        # TODO:
        lab = sitk.ReadImage(self.in_path + self.case_name)

        org_spacing = img.GetSpacing()
        org_size = img.GetSize()

        out_size = (int(org_spacing[0] * org_size[0] / target_spacing[0]),
                    int(org_spacing[1] * org_size[1] / target_spacing[1]),
                    int(org_spacing[2] * org_size[2] / target_spacing[2]))

        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(out_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(img.GetPixelIDValue())

        resampler.SetInterpolator(sitk.sitkBSpline)
        img = resampler.Execute(img)

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        lab = resampler.Execute(lab)
        return img, lab

    def augmentation(self, img, lab):
        if self.mode == 'train':
            if self.config.flip_aug:
                if np.random.random() > (1 - self.config.possibility):
                    for i in range(img.shape[0]):
                        img[i, :, :] = np.fliplr(img[i, :, :])
                        lab[i, :, :] = np.fliplr(lab[i, :, :])

            if self.config.rotation_aug:
                if np.random.random() > (1 - self.config.possibility):
                    max_angle = 10
                    angle = np.random.uniform(-max_angle, max_angle)

                    (h, w) = (img.shape[1], img.shape[2])
                    center = (h / 2, w / 2)
                    for i in range(img.shape[0]):
                        m = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img[i, :, :] = cv2.warpAffine(img[i, :, :], m, (w, h), flags=cv2.INTER_CUBIC)
                        lab[i, :, :] = cv2.warpAffine(lab[i, :, :].astype('float32'), m, (w, h),
                                                      flags=cv2.INTER_NEAREST)

            if self.config.resize_aug:
                if np.random.random() > (1 - self.config.possibility):
                    factor = 0.2 * np.random.random() + 1.0
                    img = 255 * (img / img.max())
                    img = img.astype("uint8")
                    lab = lab.astype("uint8")

                    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                    img = img.swapaxes(0, 2)
                    img = cv2.resize(img, dsize=None, fx=1, fy=factor, interpolation=cv2.INTER_CUBIC)
                    img = img.swapaxes(0, 2)

                    lab = cv2.resize(lab, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST_EXACT)
                    lab = lab.swapaxes(0, 2)
                    lab = cv2.resize(lab, dsize=None, fx=1, fy=factor, interpolation=cv2.INTER_NEAREST_EXACT).astype(
                        np.int)
                    lab = lab.swapaxes(0, 2)

            if self.config.gauss_aug:
                if np.random.random() > (1 - self.config.possibility):
                    # Todo: gauss noise sth wrong
                    noise = random_noise(img, mode='gaussian', mean=0, var=1e-3)
                    img += np.array(noise, dtype=np.float)
                    img = img / img.max()

            if self.config.gamma_aug:
                if np.random.random() > (1 - self.config.possibility):
                    mean = np.mean(255 * img)  # mean 0~255
                    gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma

                    for i in range(img.shape[0]):
                        img[i] = np.power(img[i] / img.max(), gamma_val) * 255

                    img = img / img.max()

        return img, lab

    def cut_padding(self, img, lab):
        # 未aug前考虑[128, 160, 224]
        std_size = self.config.default_size
        # cutting
        if img.shape[0] > std_size[0]:
            img = img[int(int(img.shape[0] / 2) - std_size[0] / 2): int(int(img.shape[0] / 2) + std_size[0] / 2), :, :]
            lab = lab[int(int(img.shape[0] / 2) - std_size[0] / 2): int(int(img.shape[0] / 2) + std_size[0] / 2), :, :]

        if img.shape[1] > std_size[1]:
            img = img[:, int(int(img.shape[1] / 2) - std_size[1] / 2): int(int(img.shape[1] / 2) + std_size[1] / 2), :]
            lab = lab[:, int(int(img.shape[1] / 2) - std_size[1] / 2): int(int(img.shape[1] / 2) + std_size[1] / 2), :]

        if img.shape[2] > std_size[2]:
            img = img[:, :, int(int(img.shape[2] / 2) - std_size[2] / 2): int(int(img.shape[2] / 2) + std_size[2] / 2)]
            lab = lab[:, :, int(int(img.shape[2] / 2) - std_size[2] / 2): int(int(img.shape[2] / 2) + std_size[2] / 2)]

        # padding
        padding_size = [std_size[0] - img.shape[0], std_size[1] - img.shape[1], std_size[2] - img.shape[2]]
        if any(padding_size):
            pad_d = [int((std_size[0] - img.shape[0]) / 2), std_size[0] - img.shape[0] -
                     int((std_size[0] - img.shape[0]) / 2)]
            pad_h = [int((std_size[1] - img.shape[1]) / 2), std_size[1] - img.shape[1] -
                     int((std_size[1] - img.shape[1]) / 2)]
            pad_w = [int((std_size[2] - img.shape[2]) / 2), std_size[2] - img.shape[2] -
                     int((std_size[2] - img.shape[2]) / 2)]

            img = np.pad(img, ((pad_d[0], pad_d[1]), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), mode="constant")
            lab = np.pad(lab, ((pad_d[0], pad_d[1]), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), mode="minimum")

        return img, lab

    def arr_to_tensor(self, img, lab):
        img = torch.tensor((img / img.max()).astype(np.float32), requires_grad=True).cuda().unsqueeze(0)

        lab = lab.astype(np.float32)
        lab = self.one_hot(lab)
        lab.requires_grad = True

        return img, lab

    def one_hot(self, lab):
        Lab = torch.zeros([self.config.n_classes, self.config.default_size[0], self.config.default_size[1],
                           self.config.default_size[2]]).cuda()
        for i in range(self.config.n_classes):
            Lab[i][lab == i] = 1
        return Lab

    def __getitem__(self, item):
        self.case_name = self.case_names[item]
        self.case_dir = os.path.join(self.in_path, self.case_name)
        self.case_save_dir = os.path.join(self.img_aug_saving, self.case_name)
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.case_save_dir + '_b_img.nii.gz'))
        lab = sitk.GetArrayFromImage(sitk.ReadImage(self.case_save_dir + '_0_lab.nii.gz'))
        if self.mode == "train":
            img, lab = self.augmentation(img, lab)

        img, lab = self.cut_padding(img, lab)
        img, lab = self.arr_to_tensor(img, lab)

        return self.case_name, img, lab

    def __len__(self):
        return len(self.case_names)


class Imagefolder_pre(data.Dataset):
    def __init__(self, mode, config):
        self.mode, self.config = mode, config
        self.in_path = config.img_aug_saving
        self.case_names = []
        if mode == 'train':
            for i in range(config.train_idx.shape[0]):
                self.case_names.append(config.k_fold_lists[config.train_idx[i]])

        elif mode == 'valid':
            for i in range(config.valid_idx.shape[0]):
                self.case_names.append(config.k_fold_lists[config.valid_idx[i]])

        elif mode == 'run':
            self.case_names = list(config.k_fold_lists)

        print('For {} case counts for {} in {}'.format(mode, len(self.case_names), self.in_path))

    def spacing(self):
        resampler = sitk.ResampleImageFilter()
        target_spacing = [3.2, 1.6, 1.6]

        img = sitk.ReadImage(self.in_path + self.case_name)
        # TODO:
        lab = sitk.ReadImage(self.in_path + self.case_name)

        org_spacing = img.GetSpacing()
        org_size = img.GetSize()
        print(self.case_name, org_size, org_spacing)

        out_size = (int(org_spacing[0] * org_size[0] / target_spacing[0]),
                    int(org_spacing[1] * org_size[1] / target_spacing[1]),
                    int(org_spacing[2] * org_size[2] / target_spacing[2]))

        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(out_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(img.GetPixelIDValue())

        resampler.SetInterpolator(sitk.sitkBSpline)
        img = resampler.Execute(img)

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        lab = resampler.Execute(lab)
        return img, lab

    def augmentation(self, img):
        """aug only when in train mode; input np shape: d, h, w; x varies from 0~255"""
        if self.mode == 'train':
            if self.config.flip_aug:
                if np.random.random() > (1 - self.config.possibility):
                    for i in range(img.shape[0]):
                        img[i, :, :] = np.fliplr(img[i, :, :])

            if self.config.rotation_aug:
                if np.random.random() > (1 - self.config.possibility):
                    max_angle = 10
                    angle = np.random.uniform(-max_angle, max_angle)

                    (h, w) = (img.shape[1], img.shape[2])
                    center = (h / 2, w / 2)
                    for i in range(img.shape[0]):
                        m = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img[i, :, :] = cv2.warpAffine(img[i, :, :], m, (w, h), flags=cv2.INTER_CUBIC)

            if self.config.resize_aug:
                if np.random.random() > (1 - self.config.possibility):
                    factor = 0.2 * np.random.random() + 1.0
                    img = 255 * (img / img.max())
                    img = img.astype("uint8")

                    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                    img = img.swapaxes(0, 2)
                    img = cv2.resize(img, dsize=None, fx=1, fy=factor, interpolation=cv2.INTER_CUBIC)
                    img = img.swapaxes(0, 2)

            if self.config.gauss_aug:
                if np.random.random() > (1 - self.config.possibility):
                    # Todo: gauss noise sth wrong
                    noise = random_noise(img, mode='gaussian', mean=0, var=1e-3)
                    img += np.array(noise, dtype=np.float)
                    img = img / img.max()

            if self.config.gamma_aug:
                if np.random.random() > (1 - self.config.possibility):
                    mean = np.mean(255 * img)  # mean 0~255
                    gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma

                    for i in range(img.shape[0]):
                        img[i] = np.power(img[i] / img.max(), gamma_val) * 255

                    img = img / img.max()

        return img

    def cut_padding(self, img):
        # 未aug前考虑[128, 160, 224]
        std_size = self.config.default_size
        # cutting
        if img.shape[0] > std_size[0]:
            img = img[int(int(img.shape[0] / 2) - std_size[0] / 2): int(int(img.shape[0] / 2) + std_size[0] / 2), :, :]

        if img.shape[1] > std_size[1]:
            img = img[:, int(int(img.shape[1] / 2) - std_size[1] / 2): int(int(img.shape[1] / 2) + std_size[1] / 2), :]

        if img.shape[2] > std_size[2]:
            img = img[:, :, int(int(img.shape[2] / 2) - std_size[2] / 2): int(int(img.shape[2] / 2) + std_size[2] / 2)]

        # padding
        padding_size = [std_size[0] - img.shape[0], std_size[1] - img.shape[1], std_size[2] - img.shape[2]]
        if any(padding_size):
            pad_d = [int((std_size[0] - img.shape[0]) / 2), std_size[0] - img.shape[0] -
                     int((std_size[0] - img.shape[0]) / 2)]
            pad_h = [int((std_size[1] - img.shape[1]) / 2), std_size[1] - img.shape[1] -
                     int((std_size[1] - img.shape[1]) / 2)]
            pad_w = [int((std_size[2] - img.shape[2]) / 2), std_size[2] - img.shape[2] -
                     int((std_size[2] - img.shape[2]) / 2)]

            img = np.pad(img, ((pad_d[0], pad_d[1]), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), mode="constant")

        return img

    def arr_to_tensor(self, img):
        img = torch.tensor((img / img.max()).astype(np.float32), requires_grad=True).cuda().unsqueeze(0)

        return img

    def one_hot(self, lab):
        Lab = torch.zeros([self.config.n_classes, self.config.default_size[0], self.config.default_size[1],
                           self.config.default_size[2]]).cuda()
        for i in range(self.config.n_classes):
            Lab[i][lab == i + 1] = 1
        return Lab

    def __getitem__(self, item):
        self.case_name = self.case_names[item]
        self.case_dir = os.path.join(self.in_path, self.case_name)
        # self.test_aug = os.path.join(self.config.test_aug, self.case_name)

        img = sitk.GetArrayFromImage(sitk.ReadImage(self.case_dir))

        if self.mode == "train":
            img = self.augmentation(img)

        img = self.cut_padding(img)
        img = self.arr_to_tensor(img)

        return self.case_name, img, img

    def __len__(self):
        return len(self.case_names)