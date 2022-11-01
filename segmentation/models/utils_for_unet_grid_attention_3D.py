import torch
import torch.nn as nn
import torch.nn.functional as F
from found.structure.Attention_Gated_Networks_master.models.networks_other import init_weights


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.LeakyReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.LeakyReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.LeakyReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.LeakyReLU(),)

        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetConv3_t(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1)):
        super(UnetConv3_t, self).__init__()

        # self.conv0 = nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size)
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.LeakyReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.LeakyReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),)

        self.activation = nn.LeakyReLU()
        self.conv00 = nn.Conv3d(in_size, out_size, 1, 1, 0)
        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs00 = self.conv00(inputs)
        outputs0 = self.conv1(inputs)
        outputs = self.conv2(outputs0)
        return self.activation(outputs00 + outputs)


class UnetConv3_T(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1)):
        super(UnetConv3_T, self).__init__()

        # self.conv0 = nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size)
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.LeakyReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.LeakyReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),)

        self.activation = nn.LeakyReLU()
        self.conv00 = nn.Conv3d(in_size, out_size, 1, 1, 0)
        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs00 = self.conv00(inputs)
        outputs0 = self.conv1(inputs)
        outputs = self.conv2(outputs0)
        return outputs0, self.activation(outputs00 + outputs)


class UnetConv_t(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1)):
        super(UnetConv_t, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.LeakyReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.LeakyReLU(),)

        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)

        return outputs


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.LeakyReLU(),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.InstanceNorm3d(out_size),
                                       nn.LeakyReLU(),
                                       )

        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.functional.interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        # # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('UnetConv3') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3_t(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3_t, self).__init__()
        if is_deconv:
            self.conv = UnetConv3_T(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            self.conv = UnetConv3_T(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.functional.interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        # # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('UnetConv3') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
