import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py


class FocalLoss2d(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma, phase, alpha=1, average='mean'):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.phase = phase
        self.alpha = alpha
        self.average = average

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        # assert len(input.shape) == len(target.shape)
        # assert input.size(0) == target.size(0)
        # assert input.size(1) == target.size(1)
        if self.phase == 1:
            loss = nn.BCEWithLogitsLoss()
            logpt = -loss(input, target)

            # weight = Variable(self.weight)
            # compute the negative likelyhood
            # logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
            # logpt = - nn.BCEWithLogitsLoss(input, target)

            pt = torch.exp(logpt)
            # compute the loss
            focal_loss = -( (1-pt)**self.gamma ) * logpt
            # balanced_focal_loss = self.balance_param * focal_loss
        elif self.phase == 2:
            # op = nn.LogSoftmax()
            input = F.softmax(input, dim=1)
            loss = nn.CrossEntropyLoss()
            # Target = torch.argmax(target.squeeze(0), dim=0).unsqueeze(0)
            target[0][0][target[0][0] == 1] = 0
            target[0][1][target[0][1] == 1] = 1
            target[0][2][target[0][2] == 1] = 2

            target = (target[0][0] + target[0][1] + target[0][2]).unsqueeze(0).type(torch.long)
            logpt = -loss(input, target)

            pt = torch.exp(logpt)
            focal_loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.average == 'mean':
            return focal_loss.mean()
        elif self.average == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
