#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk


class TestLoss(nn.Module):
    def __init__(self, phase=2, weight=None, ignore_index=None, average='mean', **kwargs):
        super(TestLoss, self).__init__()
        self.kwargs = kwargs
        self.phase = phase
        # TODO: weight禁止外传 当前改为活动
        # self.weight = weights

        self.ignore_index = ignore_index
        self.average = average

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        self.weight = Variable(torch.as_tensor([1 - target[0, 0].sum() / target.sum(),
                                                3 * (1 - target[0, 1].sum() / target.sum()) ** 2,
                                                10 * (1 - target[0, 2].sum() / target.sum()) ** 2]))
        test = BinaryTestLoss(phase=self.phase)
        # dice = New_weighted_BinaryDiceLoss(weight=self.weight, phase=self.phase)

        # Todo: changed code
        test_loss = 0
        predict_softmax = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = test(predict_softmax[:, i], target[:, i])
                if self.weight is not None:
                    dice_loss *= self.weight[i]
                test_loss += dice_loss

        if self.average == 'mean':
            return test_loss.mean()
        elif self.average == 'sum':
            return test_loss.sum()
        else:
            return test_loss


class BinaryTestLoss(nn.Module):
    def __init__(self, phase=1, smooth=1e-12, p=2, reduction='mean'):
        super(BinaryTestLoss, self).__init__()
        self.phase = phase
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0], "predict & target batch size don't match"

        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(pred, target)) * 2 + self.smooth
        den = num + torch.sum(torch.mul(pred[pred < 0.5], target[pred < 0.5])) + \
              torch.sum(torch.mul(pred[pred > 0.5], 1 - target[pred > 0.5])) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        weight: A float num for the weight of positive samples, (1 - weight) for negative samples
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, phase=3, smooth=1e-12, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.phase = phase
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0], "predict & target batch size don't match"
        if self.phase == 1:
            pred = torch.sigmoid(pred)

        Pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(Pred, target)) * 2 + self.smooth
        den = torch.sum(Pred.pow(self.p) + target.pow(self.p)) + self.smooth

        loss = 10 * (1 - num / den)
        # loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class New_weighted_BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        weight: A float num for the weight of positive samples, (1 - weight) for negative samples
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, phase=1, weight=0.5, smooth=1e-8, p=2, reduction='mean'):
        super(New_weighted_BinaryDiceLoss, self).__init__()
        self.phase = phase
        self.weight = weight
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict_b, predict_f, target_b, target_f, weight_b, weight_f):
        assert predict_b.shape[0] == target_b.shape[0], "predict & target batch size don't match"

        predict = predict_b.contiguous().view(predict_b.shape[0], -1)
        target = target_b.contiguous().view(target_b.shape[0], -1)

        num_b = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den_b = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        predict = predict_f.contiguous().view(predict_f.shape[0], -1)
        target = target_f.contiguous().view(target_f.shape[0], -1)

        num_f = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den_f = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = (weight_b / (weight_b + weight_f)) * (1 - num_b / den_b) + (weight_f / (weight_b + weight_f)) * (
                    1 - num_f / den_f)

        if self.reduction == 'mean':
            return 10 * loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, phase=3, weight=None, ignore_index=None, average='mean', **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.phase = phase
        # TODO: weight禁止外传 当前改为活动
        self.weight = Variable(torch.as_tensor(weight))

        self.ignore_index = ignore_index
        self.average = average
        self.partial_loss = [0, 0, 0]

    def forward(self, predict, target):
        if self.phase == 2 or self.phase == 3:
            assert predict.shape == target.shape, 'predict & target shape do not match'

            dice = BinaryDiceLoss(phase=self.phase)
            total_loss = 0
            predict_softmax = F.softmax(predict, dim=1)

            for i in range(target.shape[1]):
                if i != self.ignore_index:
                    dice_loss = dice(predict_softmax[:, i], target[:, i])
                    if self.weight is not None:
                        dice_loss *= self.weight[i]
                    self.partial_loss[i] = dice_loss
                    total_loss += dice_loss
        else:
            assert predict.shape == target.shape, 'predict & target shape do not match'
            dice = BinaryDiceLoss(phase=self.phase)

            dice_loss = dice(predict, target)

            total_loss = dice_loss

        if self.average == 'mean':
            return total_loss.mean(), self.partial_loss[0].mean(), self.partial_loss[1].mean(), self.partial_loss[2].mean()
        elif self.average == 'sum':
            return total_loss.sum(), self.partial_loss[0].sum(), self.partial_loss[1].sum(), self.partial_loss[2].sum()
        else:
            return total_loss, self.partial_loss[0], self.partial_loss[1], self.partial_loss[2]


class FocalLoss2d(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma, phase, weight, smmoth=1e-8, alpha=1, average='mean', reduction='mean', ignore_index=-100):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.phase = phase

        self.weight = Variable(torch.Tensor(weight))
        # self.balance_param = self.weight

        self.alpha = alpha
        self.average = average
        self.smooth = smmoth
        self.eps = 1
        # new test version from elektronn_loss.py
        self.nll = torch.nn.NLLLoss(weight=torch.as_tensor(weight).type(torch.float).cuda(), reduction=reduction,
                                    ignore_index=ignore_index)
        self.log_softmax = torch.nn.LogSoftmax(1)

    def forward(self, input, target):
        if self.phase == 1:
            loss = nn.BCEWithLogitsLoss()
            logpt = -loss(input, target)

            # weight = Variable(self.weight)
            # compute the negative likelyhood
            # logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
            # logpt = - nn.BCEWithLogitsLoss(input, target)

            pt = torch.exp(logpt)
            # compute the loss
            focal_loss = -((1 - pt) ** self.gamma) * logpt
        elif self.phase == 2 or self.phase == 3:
            pt = F.softmax(input, dim=1)
            n_classes = input.shape[1]
            focal_loss = 0
            # ce_loss = CELoss(self.weight, softmax=False)
            # loss = nn.BCEWithLogitsLoss()

            for i in range(n_classes):
                loss_y1 = - self.weight[i] * target[0, i] * (1 - pt[0, i]) ** self.gamma * torch.log(pt[0, i] +
                                                                                                     self.smooth) / (
                                      target[0, i].sum() + self.eps)
                #     # loss_y1 = self.weight[i] * (1 - pt[0, i]) ** self.gamma * ce_loss(pt[0, i], target[0, i])
                #     loss_y1 = - self.weight[i] * target[0, i] * (1 - pt[0, i]) ** self.gamma * torch.log(pt[0, i] + self.smooth).mean()
                #
                #     # loss_y1 = self.weight[i] * target[0, i] * (1 - pt[0, i]) ** self.gamma * ce_loss(pt[0, i], target[0, i])
                #
                focal_loss += loss_y1
            # weight = Variable(self.weight)

            # Labels = target.clone()
            # Labels[0][0][target[0][0] == 1] = 0
            # Labels[0][1][target[0][1] == 1] = 1
            # Labels[0][2][target[0][2] == 1] = 2
            #
            # Labels = (Labels[0][0] + Labels[0][1] + Labels[0][2]).unsqueeze(0).type(torch.long)
            #
            # logpt = F.cross_entropy(pt, Labels, weight=weight.cuda(), reduction=self.reduction)
            #
            # focal_loss = ((1 - torch.exp(logpt)) ** self.gamma) * logpt
            #
            # balanced_focal_loss = self.balance_param.cuda() * focal_loss

            # for i in range(n_classes):
            #
            #     # compute the negative likelyhood
            #     # logpt = - F.binary_cross_entropy_with_logits(input[0, i], target[0, i], pos_weight=weight[i],
            #     #                                              reduction=self.reduction)
            #     logpt = F.cross_entropy(pt[0, i], target[0, i], weight=weight[i], reduction=self.reduction)
            #     pt = torch.exp(logpt)
            #
            #     # compute the loss
            #     focal_loss += -((1 - pt) ** self.gamma) * logpt

            # return balanced_focal_loss

            # log_prob = self.log_softmax(input)
            # prob = torch.exp(log_prob)
            # prob = F.softmax(input, dim=1)
            #
            # Labels = target.clone()
            # Labels[0][0][target[0][0] == 1] = 0
            # Labels[0][1][target[0][1] == 1] = 1
            # Labels[0][2][target[0][2] == 1] = 2
            #
            # Labels = (Labels[0][0] + Labels[0][1] + Labels[0][2]).unsqueeze(0).type(torch.long)
            #
            # focal_loss = 0
            # c = target.shape[1]
            # for i in range(c):
            #     fg = (Labels[0][i] == 1).float()
            #     class_pred = prob[:, c]
            #
            #     focal_loss += - self.weight[i] * fg * (1 - class_pred) ** self.gamma * torch.log(class_pred + self.smooth)
            # focal_loss = self.nll(((1 - prob) ** self.gamma) * log_prob, Labels)

        if self.average == 'mean':
            return focal_loss.mean()
        elif self.average == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CELoss(nn.Module):
    def __init__(self, weight, softmax=True, smooth=1e-8, average='mean'):
        super(CELoss, self).__init__()

        self.weight = torch.Tensor(weight)
        self.smooth = smooth
        self.softmax = softmax
        self.average = average

    def forward(self, pred, gt):
        ce_loss1, ce_loss = 0, 0

        if self.softmax:
            pt = F.softmax(pred, dim=1)
            n_classes = pred.shape[1]
            for i in range(n_classes):
                loss_y1 = - (self.weight[i] / self.weight.sum()) * gt[0, i] * torch.log(pt[0, i] + self.smooth)

                ce_loss += loss_y1
        else:
            # pt = F.softmax(pred)

            # pt = pred.clone()
            loss_y1 = - gt * torch.log(pred + self.smooth)
            ce_loss = loss_y1

        if self.average == 'mean':
            return ce_loss.mean()
        elif self.average == 'sum':
            return ce_loss.sum()
        else:
            return ce_loss


"""
https://github.com/bermanmaxim/LovaszSoftmax
Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from torch.autograd import Variable
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------

class lovasz_softmax(nn.Module):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """

    def __init__(self, weight, classes='present', per_image=False, ignore=None):
        super(lovasz_softmax, self).__init__()

        self.weight = torch.Tensor(weight)
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        if self.per_image:
            loss = mean(
                lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore),
                                    classes=self.classes)
                for prob, lab in zip(probas, labels))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, self.ignore), self.weight, classes=self.classes)

        return loss


# def lovasz_softmax(probas, labels, weight, classes='present', per_image=False, ignore=None):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
#               Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
#       labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
#       classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#       per_image: compute the loss per image instead of per batch
#       ignore: void class labels
#     """
#     if per_image:
#         loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
#                           for prob, lab in zip(probas, labels))
#     else:
#         loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), weight, classes=classes)
#
#     return loss


def lovasz_softmax_flat(probas, labels, weight, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]

        # fc_loss = FocalLoss2d(gamma=5, phase=1, weight=torch.as_tensor([1.0, 3.0, 3.0]), average='None')
        # fc_loss_part = fc_loss(class_pred, fg)
        #
        # dc_loss = DiceLoss(phase=1, weight=torch.as_tensor([1.0, 3.0, 3.0]), average='None')
        # dc_loss_part = dc_loss(class_pred, fg)

        # errors = (Variable(fg) - class_pred).abs()
        ce_loss = CELoss(weight=[1, 1, 1], softmax=False, average='None')
        errors = ce_loss(class_pred, fg)
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]

        # Original
        # losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))

        # New test version
        # class_pred_sorted = class_pred[perm]
        # print(0.5 * (torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))) + 10 * fc_loss_part).item(),
        #       dc_loss_part.item())
        # losses.append(0.5 * (torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))) + fc_loss_part))
        losses.append(weight[c] * torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))

    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        B, C, D, H, W = probas.size()
        probas = F.softmax(probas, dim=1)
        probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

    # New test version from elektroon_lovasz_loss.py
    Labels = labels.clone()
    Labels[0][0][labels[0][0] == 1] = 0
    Labels[0][1][labels[0][1] == 1] = 1
    Labels[0][2][labels[0][2] == 1] = 2
    #
    # # image = (labels[0][0] + labels[0][1] + labels[0][2]).cpu().numpy().astype(np.float32)
    # # image = sitk.GetImageFromArray(image)
    # # sitk.WriteImage(image, '/data0/lyx/kits_data/debug_data_saving/labels.nii.gz')
    #
    Labels = (Labels[0][0] + Labels[0][1] + Labels[0][2]).unsqueeze(0).type(torch.long)
    # Labels = Labels.view(-1)
    Labels = Labels.view(-1)

    if ignore is None:
        return probas, Labels

    valid = (Labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = Labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
