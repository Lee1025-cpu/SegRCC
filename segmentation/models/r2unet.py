import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Block for convolutional layer of U-Net at the encoder end.
    Args:
        ch_in : number of input channels
        ch_out : number of outut channels
    Returns:
        feature map of the giv
    """

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Block for deconvolutional layer of U-Net at the decoder end
    Args:
        ch_in : number of input channels
        ch_out : number of outut channels
    Returns:
        feature map of the given input
    """

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    """
    Recurrent convolution block for RU-Net and R2U-Net
    Args:
        ch_out : number of outut channels
        t: the number of recurrent convolution block to be used
    Returns:
        feature map of the given input
    """

    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1 = self.conv(x)
        for i in range(1, self.t):
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    """
    Recurrent Residual convolution block for R2U-Net
    Args:
        ch_in  : number of input channels
        ch_out : number of outut channels
        t	: the number of recurrent residual convolution block to be used
    Returns:
        feature map of the given input
    """

    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1  # residual learning


class RCNN_block(nn.Module):
    """
    Recurrent convolution block for RU-Net
    Args:
        ch_in  : number of input channels
        ch_out : number of outut channels
        t	: the number of recurrent residual convolution block to be used
    Returns:
        feature map of the given input
    """

    def __init__(self, ch_in, ch_out, t=2):
        super(RCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.RCNN(x)
        return x


class ResCNN_block(nn.Module):
    """
    Residual convolution block
    Args:
        ch_in  : number of input channels
        ch_out : number of outut channels
    Returns:
        feature map of the given input
    """

    def __init__(self, ch_in, ch_out):
        super(ResCNN_block, self).__init__()
        self.Conv = conv_block(ch_in, ch_out)
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv_1x1(x)
        x = self.Conv(x)
        return x + x1


class R2U_Net(nn.Module):
    """
    R2U-Net Network.
    Implements U-Net with a RRCNN block.

    Args:
        img_ch: Input image channels
        output_ch: Number of channels expected in the output
        t: number of recurrent blocks expected

    Returns:
        Feature map of input (batch_size, output_ch=1,h,w)
    """

    def __init__(self, img_ch=3, output_ch=3, t=2, feature_scale=32, dropout=0.3):
        super(R2U_Net, self).__init__()

        filters = [1, 2, 4, 8, 16]
        filters = [int(x * feature_scale) for x in filters]

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=filters[0], t=t)

        self.RRCNN2 = RRCNN_block(ch_in=filters[0], ch_out=filters[1], t=t)

        self.RRCNN3 = RRCNN_block(ch_in=filters[1], ch_out=filters[2], t=t)

        self.RRCNN4 = RRCNN_block(ch_in=filters[2], ch_out=filters[3], t=t)

        self.RRCNN5 = RRCNN_block(ch_in=filters[3], ch_out=filters[4], t=t)

        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Up_RRCNN5 = RRCNN_block(ch_in=filters[4], ch_out=filters[3], t=t)

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Up_RRCNN4 = RRCNN_block(ch_in=filters[3], ch_out=filters[2], t=t)

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Up_RRCNN3 = RRCNN_block(ch_in=filters[2], ch_out=filters[1], t=t)

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Up_RRCNN2 = RRCNN_block(ch_in=filters[1], ch_out=filters[0], t=t)

        self.Conv_1x1 = nn.Conv3d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()
        self.act1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout3d(p=dropout)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        x2 = self.Maxpool(x1)

        x2 = self.RRCNN2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.RRCNN3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.RRCNN4(x4)
        x5 = self.Maxpool(x4)

        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)

        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)

        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)

        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)

        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        # d1 = self.act(d1)
        d1 = self.dropout(d1)
        return d1
