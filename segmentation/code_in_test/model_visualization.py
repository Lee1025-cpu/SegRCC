# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-05-31 15:37
# 文件名称：model_visualization
# 开发工具：PyCharm
import os
import torch
from torchsummary import summary
from tensorboardX import SummaryWriter
from Now_state.segmentation.models.munet import Modified3DUNet

# TODO: model visualization
net = Modified3DUNet(in_channels=1, n_classes=2, base_n_filter=3, dropout=0.2)
# 1st: print different layers involved and their specifications.
print(net)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.set_num_threads(8)

net.cuda()
# 2nd: torchsummary model
# must take the input size and batch size is set to -1 meaning any batch size we provide.
summary(net, input_size=(1, 128, 128, 128))

# 3rd: summarywriter---tensorboard
x = torch.rand((1, 1, 128, 128, 128))
with SummaryWriter(log_dir='/data0/lyx/Code_grad/net_architecture_visualization/BraTs18_munet',
                   comment='munet') as w:
    w.add_graph(net, x)  # 这其实和tensorflow里面的summarywriter是一样的
