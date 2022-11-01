# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-05-31 15:56
# 文件名称：make_dot_test
# 开发工具：PyCharm

import hickle as hkl
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchviz import make_dot


# params = hkl.load('resnet-18-at-export.hkl')

# ValueError: Provided argument 'file_obj' does not appear to be a valid hickle file!
# (HDF5-file does not have the proper attributes!)
params = hkl.load('F:/googledownload/resnet-18-at-export.hkl')

# # convert numpy arrays to torch Variables
# for k, v in sorted(params.items()):
#     print(k, tuple(v.shape))
#     params[k] = Variable(torch.from_numpy(v), requires_grad=True)
#
# print('\nTotal parameters:', sum(v.numel() for v in params.values()))