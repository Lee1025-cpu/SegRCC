# _*_ coding:utf-8 _*_
# Lee 2022-02-17 10:20 test
# Note: 

import os
import torch
import numpy as np


def softmax_torch(x):
    # Assuming x has atleast 2 dimensions
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    probs = x_exp/x_exp_sum
    return probs


def softmax_np(x):
    maxes = np.max(x, axis=1, keepdims=True)[0]
    x_exp = np.exp(x-maxes)
    x_exp_sum = np.sum(x_exp, 1, keepdims=True)
    probs = x_exp/x_exp_sum
    return probs


def test1():
    if __name__ == "__main__":
        x = torch.randn(1, 3, 128, 128, 128)
        std_pytorch_softmax = torch.nn.functional.softmax(x)
        pytorch_impl = softmax_torch(x)
        numpy_impl = softmax_np(x.detach().cpu().numpy())
        print("Shapes: x --> {}, std --> {}, pytorch impl --> {}, numpy impl --> {}".format(x.shape,
                                                                                            std_pytorch_softmax.shape,
                                                                                            pytorch_impl.shape,
                                                                                            numpy_impl.shape))
        print("Std and torch implementation are same?", torch.allclose(std_pytorch_softmax, pytorch_impl))
        print("Std and numpy implementation are same?",
              torch.allclose(std_pytorch_softmax, torch.from_numpy(numpy_impl)))


if __name__ == "__main__":
    test1()
    from torchvision import models
