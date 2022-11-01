# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-01-18 10:47
# 文件名称：about_scatter_
# 开发工具：PyCharm
import torch
import numpy as np


if __name__=="__main__":
    src = 1
    input_tensor = torch.zeros(3, 3)
    index_tensor = torch.tensor([[0, 1, 2], [2, 0, 1], [0, 2, 1]])
    print(index_tensor.scatter_(0, index_tensor, src))

    target = torch.randint(0, 10, (10,))
    print(target)
    one_hot = torch.nn.functional.one_hot(target)
    print(one_hot)