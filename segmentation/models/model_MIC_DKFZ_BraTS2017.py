import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torchsummary import summary
from torch.autograd import gradcheck


# adapt from https://github.com/MIC-DKFZ/BraTS2017


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        # Todo: affine set to true, initialized the same way as done for batch normalization
        # m = nn.InstanceNorm3d(planes)
        m = nn.InstanceNorm3d(planes, affine=True)

    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='in', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv1_0 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn1_0   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        # TODO: changed 0827
        # x = self.bn1(self.conv1(x))
        x = self.bn1_0(self.conv1_0(self.relu(self.bn1(self.conv1(x)))))

        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes, norm='in', first=False):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.relu = nn.LeakyReLU()

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))

        y = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
        y = self.relu(self.bn2(self.conv2(y)))

        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y


class Unet(nn.Module):
    def __init__(self, c=1, n=3, dropout=0.3, norm='in', num_classes=1):
        super(Unet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD(c,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.convu4 = ConvU(16*n, norm, True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y3 = self.seg3(y3)
        y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) + self.upsample(y2)

        return y1


if __name__ == "__main__":
    # import torch
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # cuda0 = torch.device('cuda:0')
    # x = torch.rand((2, 4, 32, 32, 32), device=cuda0)
    # model = Unet()
    # model.cuda()
    # y = model(x)
    # print(y.shape)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    torch.set_num_threads(8)
    #
    # import torch
    #
    model = Unet(c=1, num_classes=3, n=4, dropout=0.3)
    # # model.cuda()
    # # summary(model, input_size=(1, 128, 128, 128))
    #
    from tensorboardX import SummaryWriter

    x = torch.rand((1, 1, 128, 128, 128))

    with SummaryWriter(log_dir='/data0/lyx/Code_grad/net_architecture_visualization/model_MIC_DKFZ_BraTS2017',
                       comment='model_MIC_DKFZ_BraTS2017') as w:
        w.add_graph(model, x)  # 这其实和tensorflow里面的summarywriter是一样的。
    #
    # # print(model)

    # net = Unet().cuda().double()
    # inputs = torch.randn((1, 1, 32, 32, 32), requires_grad=True, dtype=torch.double).cuda()
    #
    # test = gradcheck(net, inputs)
    # print("Are the gradients correct: ", test)

