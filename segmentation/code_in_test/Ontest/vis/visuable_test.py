# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-04-21 11:19
# 文件名称：visuable_test
# 开发工具：PyCharm
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
from torchvision.datasets import ImageFolder

# torch.cuda.set_device(0)  # 设置GPU ID
# is_cuda = True
# simple_transform = transforms.Compose([transforms.Resize((224, 224)),
#                                        transforms.ToTensor(),  # H, W, C -> C, W, H 归一化到(0,1)，简单直接除以255
#                                        transforms.Normalize([0.485, 0.456, 0.406],  # std
#                                                             [0.229, 0.224, 0.225])])
#
# # mean  先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
# # 使用 ImageFolder 必须有对应的目录结构
# train = ImageFolder("./datas/dogs-vs-cats/train", simple_transform)
# valid = ImageFolder("./datas/dogs-vs-cats/valid", simple_transform)
# train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=5)
# val_loader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=5)
#
# vgg = models.vgg16(pretrained=True).cuda()
#
#
# # 提取不同层输出的 主要代码
# class LayerActivations:
#     features = None
#
#     def __init__(self, model, layer_num):
#         self.hook = model[layer_num].register_forward_hook(self.hook_fn)
#
#     def hook_fn(self, module, input, output):
#         self.features = output.cpu()
#
#     def remove(self):
#         self.hook.remove()
#
#
# print(vgg.features)
#
# conv_out = LayerActivations(vgg.features, 0)  # 提出第 一个卷积层的输出
# img = next(iter(train_loader))[0]
#
# # imshow(img)
# o = vgg(Variable(img.cuda()))
# conv_out.remove()  #
# act = conv_out.features  # act 即 第0层输出的特征
#
# # 可视化 输出
# fig = plt.figure(figsize=(20, 50))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
# for i in range(30):
#     ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
#     ax.imshow(act[0][i].detach().numpy(), cmap="gray")
#
# plt.show()

import torch
import torch.nn as nn


class TestForHook(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.linear_2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.initialize()

    def forward(self, x):
        linear_1 = self.linear_1(x)
        linear_2 = self.linear_2(linear_1)
        relu = self.relu(linear_2)
        relu_6 = self.relu6(relu)
        layers_in = (x, linear_1, linear_2)
        layers_out = (linear_1, linear_2, relu)
        return relu_6, layers_in, layers_out

    def initialize(self):
        """ 定义特殊的初始化，用于验证是不是获取了权重"""
        self.linear_1.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1], [1, 1]]))
        self.linear_1.bias = torch.nn.Parameter(torch.FloatTensor([1, 1]))
        self.linear_2.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1]]))
        self.linear_2.bias = torch.nn.Parameter(torch.FloatTensor([1]))
        return True

# 1：定义用于获取网络各层输入输出tensor的容器
# 并定义module_name用于记录相应的module名字
module_name = []
features_in_hook = []
features_out_hook = []


# 2：hook函数负责将获取的输入输出添加到feature列表中
# 并提供相应的module名字
def hook(module, fea_in, fea_out):
    print("hooker working")
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

# 3：定义全部是1的输入
x = torch.FloatTensor([[0.1, 0.1], [0.1, 0.1]])

# 4:注册钩子可以对某些层单独进行
net = TestForHook()
net_chilren = net.children()
for child in net_chilren:
    if not isinstance(child, nn.ReLU6):
        child.register_forward_hook(hook=hook)

# 5:测试网络输出
out, features_in_forward, features_out_forward = net(x)
print("*"*5+"forward return features"+"*"*5)
print(features_in_forward)
print(features_out_forward)
print("*"*5+"forward return features"+"*"*5)


# 6:测试features_in是不是存储了输入
print("*"*5+"hook record features"+"*"*5)
print(features_in_hook)
print(features_out_hook)
print(module_name)
print("*"*5+"hook record features"+"*"*5)

# 7：测试forward返回的feautes_in是不是和hook记录的一致
print("sub result")
for forward_return, hook_record in zip(features_in_forward, features_in_hook):
    print(forward_return-hook_record[0])