# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2020-12-22 16:01
# 文件名称：model
# 开发工具：PyCharm
import torch
import torch.nn as nn


class ConvBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, k_size=3, stride=1, padding=1):
        super(ConvBlockDown, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=k_size,
                                stride=stride, padding=padding, dilation=1)
        # self.batch_norm = nn.BatchNorm3d(num_features=out_channel)
        self.batch_norm = nn.InstanceNorm3d(num_features=out_channel)

        # self.activate = nn.ReLU(inplace=True)
        self.activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm3d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.activate(self.batch_norm(self.conv3d(x)))

        return x


class ConvBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, k_size=3, stride=1, padding=2):
        super(ConvBlockUp, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=k_size,
                                stride=stride, padding=padding, dilation=2)
        # self.batch_norm = nn.BatchNorm3d(num_features=out_channel)
        self.batch_norm = nn.InstanceNorm3d(num_features=out_channel)

        # self.activate = nn.ReLU(inplace=True)
        self.activate = nn.LeakyReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm3d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.activate(self.batch_norm(self.conv3d(x)))

        return x


class Three_D_Unet_DKFZ(nn.Module):
    """
    An attempt at beating the 3D U-Net
    https://arxiv.org/pdf/1908.02182.pdf
    Forward/backward pass size (MB): 13413.76
    Too huge
    """
    def __init__(self, in_channel=1, out_channel=1, slice_num=30):
        super(Three_D_Unet_DKFZ, self).__init__()

        self.conv1 = ConvBlockDown(in_channel, slice_num)
        self.conv2 = ConvBlockDown(slice_num, slice_num)
        self.max1 = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])

        self.conv3 = ConvBlockDown(slice_num, slice_num * 2)
        self.conv4 = ConvBlockDown(slice_num * 2, slice_num * 2)
        self.max2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = ConvBlockDown(slice_num * 2, slice_num * 4)
        self.conv6 = ConvBlockDown(slice_num * 4, slice_num * 4)
        self.max3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv7 = ConvBlockDown(slice_num * 4, slice_num * 8)
        self.conv8 = ConvBlockDown(slice_num * 8, slice_num * 8)
        self.max4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv9 = ConvBlockDown(slice_num * 8, 320)
        self.conv10 = ConvBlockDown(320, 320)
        self.max5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv11 = ConvBlockDown(320, 320)
        self.conv12 = ConvBlockDown(320, 320)

        self.up1 = nn.ConvTranspose3d(320, 320, kernel_size=2, stride=2)

        self.u_conv1 = ConvBlockUp(640, 320)
        self.u_conv2 = ConvBlockUp(320, 320)

        self.up2 = nn.ConvTranspose3d(320, slice_num * 8, kernel_size=2, stride=2)

        self.u_conv3 = ConvBlockUp(slice_num * 16, slice_num * 8)
        self.u_conv4 = ConvBlockUp(slice_num * 8, slice_num * 8)

        self.up3 = nn.ConvTranspose3d(slice_num * 8, slice_num * 4, kernel_size=2, stride=2)

        self.u_conv5 = ConvBlockUp(slice_num * 8, slice_num * 4)
        self.u_conv6 = ConvBlockUp(slice_num * 4, slice_num * 4)

        self.up4 = nn.ConvTranspose3d(slice_num * 4, slice_num * 2, kernel_size=2, stride=2)

        self.u_conv7 = ConvBlockUp(slice_num * 4, slice_num * 2)
        self.u_conv8 = ConvBlockUp(slice_num * 2, slice_num * 2)

        self.up5 = nn.ConvTranspose3d(slice_num * 2, slice_num * 1, kernel_size=[1, 2, 2], stride=[1, 2, 2])

        self.u_conv9 = ConvBlockUp(slice_num * 1, slice_num * 1)
        self.u_conv10 = ConvBlockUp(slice_num * 1, slice_num * 1)

        self.conv_last = nn.Conv3d(slice_num, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        conv2 = self.conv2(self.conv1(x))
        conv2m1 = self.max1(conv2)

        conv4 = self.conv4(self.conv3(conv2m1))
        conv4m2 = self.max2(conv4)

        conv6 = self.conv6(self.conv5(conv4m2))
        conv6m3 = self.max3(conv6)

        conv8 = self.conv8(self.conv7(conv6m3))
        conv8m4 = self.max4(conv8)

        conv10 = self.conv10(self.conv9(conv8m4))
        conv10m5 = self.max5(conv10)

        conv12 = self.conv12(self.conv11(conv10m5))

        u1 = self.up1(conv12)

        u_conv2 = self.u_conv2(self.u_conv1(torch.cat((u1, conv10), dim=1)))
        u2 = self.up2(u_conv2)

        u_conv4 = self.u_conv4(self.u_conv3(torch.cat((u2, conv8), dim=1)))
        u3 = self.up3(u_conv4)

        u_conv6 = self.u_conv6(self.u_conv5(torch.cat((u3, conv6), dim=1)))
        u4 = self.up4(u_conv6)

        u_conv8 = self.u_conv8(self.u_conv7(torch.cat((u4, conv4), dim=1)))
        u5 = self.up5(u_conv8)

        conv_last = self.conv_last(self.u_conv10(self.u_conv9(u5)))

        return conv_last


if __name__ == '__main__':
    import os
    from torchsummary import summary

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.set_num_threads(8)
    model = Three_D_Unet_DKFZ()
    model.cuda()
    summary(model, input_size=(1, 80, 160, 160))





