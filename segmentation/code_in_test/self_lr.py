# _*_ coding:utf-8 _*_
# Lee 2022-05-26 10:21 self_lr
# Note: 

import torch
from torch.optim import lr_scheduler, Adam
import matplotlib.pyplot as plt
import torch.nn as nn
import math


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc = nn.Linear(1, 10)

    def forward(self, x):
        return self.fc(x)


model = net()
lr0 = 6.73e-6
optimizer = Adam(model.parameters(), lr=lr0)

lr_list = [lr0]

lf = lambda x: (((1 + math.cos(x * math.pi / 100)) / 2) ** 1.0) * 0.8 + 0.2 if x < 100 else 0.1  # cosine


def nre_lr(x):
    if x < 45:
        return (((1 + math.cos(x * math.pi / 15)) / 2) ** 1.0) * 0.38 + 0.62
    elif 45 <= x < 50:
        return 0.62
    elif 50 <= x < 95:
        return (((1 + math.cos((x - 50) * math.pi / 15)) / 2) ** 1.0) * 1.99 + 2.60
    else:
        return 2.6


# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self_lr)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=nre_lr)

for epoch in range(100):
    optimizer.step()
    scheduler.step()
    lr = scheduler.get_lr()[0]
    lr_list.append(lr)
t = range(len(lr_list))
plt.plot(t, lr_list, color='r')
plt.show()
