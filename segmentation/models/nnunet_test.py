# _*_ coding:utf-8 _*_
# Lee 2022-02-21 15:11 nnunet_test
# Note: 

import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1, dropout=0.3, more_mode=True,
                 pre_activation=True):
        """

        :param in_size:
        :param out_size:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dropout:
        :param more_mode: If True, two blocks, else one. Default: True
        :param pre_activation: If True, instance norm + leaky_relu + conv, else, conv + instance norm + leaky_relu.
            Default: True
        """
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.more_mode = more_mode

        if pre_activation:
            self.conv1 = nn.Sequential(nn.InstanceNorm3d(in_size, affine=True),
                                       nn.LeakyReLU(),
                                       nn.Conv3d(in_size, out_size, kernel_size, stride, padding),)
            self.conv2 = nn.Sequential(nn.InstanceNorm3d(out_size, affine=True),
                                       nn.LeakyReLU(),
                                       nn.Conv3d(out_size, out_size, kernel_size, 1, padding),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, stride, padding),
                                       nn.InstanceNorm3d(out_size, affine=True),
                                       nn.LeakyReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding),
                                       nn.InstanceNorm3d(out_size, affine=True),
                                       nn.LeakyReLU(),)

    def forward(self, x):
        x1 = self.conv1(x)
        if self.more_mode:
            x1 = self.conv2(x1)
        if self.dropout > 0:
            x1 = F.dropout3d(x1, self.dropout)

        return x1


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1, dropout=0.3, more_mode=False,
                 pre_activation=True):
        """
        Refer to: Identity Mappings in Deep Residual Networks.

        :param in_size:
        :param out_size:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dropout:
        :param more_mode: If True, two blocks, else one. Default: True
        :param pre_activation: If True, instance norm + leaky_relu + conv, else, conv + instance norm + leaky_relu.
            Default: True
        """
        super(ResBlock, self).__init__()
        self.dropout = dropout
        self.more_mode = more_mode
        self.conv0 = nn.Conv3d(in_size, out_size, 1, 1, 0)

        if pre_activation:
            self.conv1 = nn.Sequential(nn.InstanceNorm3d(in_size, affine=True),
                                       nn.LeakyReLU(),
                                       nn.Conv3d(in_size, out_size, kernel_size, stride, padding),)
            self.conv2 = nn.Sequential(nn.InstanceNorm3d(out_size, affine=True),
                                       nn.LeakyReLU(),
                                       nn.Conv3d(out_size, out_size, kernel_size, 1, padding),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, stride, padding),
                                       nn.InstanceNorm3d(out_size, affine=True),
                                       nn.LeakyReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding),
                                       nn.InstanceNorm3d(out_size, affine=True),
                                       nn.LeakyReLU(),)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        if self.more_mode:
            x1 = self.conv2(x1)
        if self.dropout > 0:
            x1 = F.dropout3d(x1, self.dropout)

        return x0+x1


class Conv3(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.3, res_num=1, first=False, max_op="conv", more_mode=True,
                 pre_activation=True):
        """
        Down sample in Unet.

        :param in_size:
        :param out_size:
        :param dropout: Default: 0.3
        :param res_num: Num of the residual block. Default: 1
        :param first: Label the first in the highest resolution without max_op. Default: False
        :param max_op: 选择下采样的方式。"conv" for stride conv and "max" for max pooling. Default: "conv"
        :param more_mode: If True, two blocks, else one. Default: True
        :param pre_activation: If True, instance norm + leaky_relu + conv, else, conv + instance norm + leaky_relu.
            Default: True
        """
        super(Conv3, self).__init__()
        self.dropout = dropout
        self.res_num = res_num
        self.first = first

        if max_op == "max":
            self.max = nn.MaxPool3d(2, 2)
        elif max_op == "conv":
            self.max = nn.Conv3d(in_size, in_size, 3, 2, 1, bias=False)

        self.conv = ConvBlock(in_size, out_size, kernel_size=3, stride=1, padding=1, dropout=dropout,
                              more_mode=more_mode, pre_activation=pre_activation)
        self.res_conv = ResBlock(out_size, out_size, kernel_size=3, stride=1, padding=1, dropout=dropout,
                                 more_mode=more_mode, pre_activation=pre_activation)

    def forward(self, x):
        if not self.first:
            x = self.max(x)
        x = self.conv(x)

        for i in range(self.res_num):
            x = self.res_conv(x)

        return x


class ConvU(nn.Module):
    def __init__(self, in_size, out_size, up_sample="convt", more_mode=False):
        """
        Up sample in Unet.

        :param in_size: 
        :param out_size: 
        :param up_sample: 可选为"up", "convt", "inter"分别对应上采样，转置卷积，插值, default: "convt"
        :param more_mode: If True, two blocks, else one. Default: True
        """
        super(ConvU, self).__init__()
        if up_sample == "up":
            # TODO: check the out size
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        elif up_sample == "convt":
            self.upsample = nn.ConvTranspose3d(in_size, in_size//2, kernel_size=(4, 4, 4), stride=(2, 2, 2),
                                               padding=(1, 1, 1))
        elif up_sample == "inter":
            self.upsample = nn.functional.interpolate(scale_factor=2, mode='trilinear')

        self.conv = ConvBlock(in_size//2 + out_size, out_size, kernel_size=3, stride=1, padding=1, dropout=0.0,
                              more_mode=more_mode)

    def forward(self, x, prev):
        x = self.upsample(x)
        y = torch.cat([prev, x], 1)
        y = self.conv(y)

        return y


class nnUnet(nn.Module):
    def __init__(self, feature_num, c=1, num_classes=3, dropout=0.3, max_op="conv", up_sample="convt",
                 pre_activation=True):
        """
        An attempt at beating the 3D U-Net.的尝试性复现

        :param feature_num: channel的计划列表
        :param c: batch_size
        :param num_classes:
        :param dropout: Default: 0.3
        :param max_op: 下采样中，可选为"max", "conv"分别对应最大池化，跨步卷积（stride==2）, default: "conv"
        :param up_sample: 上采样中，可选为"up", "convt", "inter"分别对应上采样，转置卷积，插值, default: "convt"
        :param pre_activation: If True, instance norm + leaky_relu + conv, else, conv + instance norm + leaky_relu.
            Default: True
        """
        super(nnUnet, self).__init__()
        self.feature_num = feature_num

        self.convd1 = Conv3(c, self.feature_num[0], dropout, 1, first=True, max_op=max_op,
                            pre_activation=pre_activation)
        self.convd2 = Conv3(self.feature_num[0], self.feature_num[1], dropout, 1, max_op=max_op,
                            pre_activation=pre_activation)
        self.convd3 = Conv3(self.feature_num[1], self.feature_num[2], dropout, 2, max_op=max_op,
                            pre_activation=pre_activation)
        self.convd4 = Conv3(self.feature_num[2], self.feature_num[3], dropout, 3, max_op=max_op,
                            pre_activation=pre_activation)
        self.convd5 = Conv3(self.feature_num[3], self.feature_num[4], dropout, 4, max_op=max_op,
                            pre_activation=pre_activation)
        self.convd6 = Conv3(self.feature_num[4], self.feature_num[5], dropout, 5, max_op=max_op,
                            pre_activation=pre_activation)

        self.convu5 = ConvU(self.feature_num[5], self.feature_num[4], up_sample, more_mode=False)
        self.convu4 = ConvU(self.feature_num[4], self.feature_num[3], up_sample, more_mode=False)
        self.convu3 = ConvU(self.feature_num[3], self.feature_num[2], up_sample, more_mode=False)
        self.convu2 = ConvU(self.feature_num[2], self.feature_num[1], up_sample, more_mode=False)
        self.convu1 = ConvU(self.feature_num[1], self.feature_num[0], up_sample, more_mode=False)

        self.seg = nn.Conv3d(self.feature_num[0], num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)
        x6 = self.convd6(x5)

        y5 = self.convu5(x6, x5)
        y4 = self.convu4(y5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y = self.seg(y1)

        return y
