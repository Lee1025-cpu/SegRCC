# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2020-12-22 17:09
# 文件名称：evaluation
# 开发工具：PyCharm
import torch
from utility import *
import torch.nn.functional as F


def TP(pred, gt):
    return torch.sum((pred == 1) & (gt == 1)).float()


def FP(pred, gt):
    return torch.sum((pred == 1) & (gt == 0)).float()


def TN(pred, gt):
    return torch.sum((pred == 0) & (gt == 0)).float()


def FN(pred, gt):
    return torch.sum((pred == 0) & (gt == 1)).float()


def get_class_imbalance_ratio(epoch, pred, gt):
    if epoch == 0:
        num = TP(pred, gt)
        den = gt.shape[-1] * gt.shape[-2] * gt.shape[-3]
        class_imbalance_ratio = float(num / den)
        print("Class_imbalance_ratio is:", class_imbalance_ratio)


def acc(pre, gt):
    tmp = pre + gt
    a1 = torch.sum(torch.where(tmp == 2, 1, 0))
    a0 = torch.sum(torch.where(tmp == 0, 1, 0))
    Size = pre.shape[0] * pre.shape[1] * pre.shape[2]
    Acc = (a1 + a0) / Size
    return Acc.item()


def get_acc(pred, gt):
    n_classes = pred.shape[1]
    ACC = np.zeros([1, n_classes])

    for c in range(n_classes):
        ACC[0, c] = acc(pred[0][c], gt[0][c])

    return ACC.mean()


def precision(pre, gt):
    tmp = pre + gt
    a = torch.sum(torch.where(tmp == 2, 1, 0))
    c = torch.sum(gt)
    prec = a / c
    return prec.item()


def get_pc(pred, gt):
    n_classes = pred.shape[1]
    PC = np.zeros([1, n_classes])

    for c in range(n_classes):
        PC[0, c] = precision(pred[0][c], gt[0][c])

    return PC.mean()


def recall(pre, gt):
    tmp = pre + gt
    a = torch.sum(torch.where(tmp == 2, 1, 0))
    b = torch.sum(pre.to(torch.float))

    if b.item() == 0:
        rec = torch.tensor(0).cuda()
    else:
        rec = a / b
    return rec.item()


def get_recall(pred, gt):
    n_classes = pred.shape[1]
    REC = np.zeros([1, n_classes])

    for c in range(n_classes):
        REC[0, c] = recall(pred[0][c], gt[0][c])

    return REC.mean()


def dice(pre, gt):
    tmp = pre + gt
    a = torch.sum(torch.where(tmp == 2, 1, 0))
    b = torch.sum(pre.to(torch.float))
    c = torch.sum(gt)
    dice = (2 * a) / (b + c)
    return dice.item()


def get_dc(pred, gt):
    n_classes = pred.shape[1]
    DSC = np.zeros([1, n_classes])

    for c in range(n_classes):
        DSC[0, c] = dice(pred[0][c], gt[0][c])

    # return DSC.mean(), DSC[0, 0], DSC[0, 1]
    return DSC.mean(), DSC[0, 1], DSC[0, 2]


def evaluation(pred, gt, loss, loss_bk, loss_kidney, loss_tumor, Id, log, eva):
    # ["id", "loss", "loss_bk", "loss_kid", "loss_tu", "dc", "pc", "recall", "dc_kid", "dc_tumor", "acc"]
    loss = loss.item()
    id = Id[0]

    n_classes = pred.shape[1]
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    Pred = torch.zeros([pred.shape[0], n_classes, pred.shape[1], pred.shape[2], pred.shape[3]]).cuda()
    for i in range(n_classes):
        Pred[0][i][pred[0] == i] = 1
    # n_classes = pred.shape[1]
    # pred = torch.argmax(pred, dim=1)
    # Pred = torch.zeros([pred.shape[0], n_classes, pred.shape[1], pred.shape[2], pred.shape[3]]).cuda()
    # for i in range(n_classes):
    #     Pred[0][i][pred[0] == i] = 1

    dc, dc_kid, dc_tumor = get_dc(Pred, gt)
    pc = get_pc(Pred, gt)
    recall = get_recall(Pred, gt)
    acc = get_acc(Pred, gt)

    new_data = [id, loss, loss_bk.item(), loss_kidney.item(), loss_tumor.item(), dc, pc, recall, dc_kid, dc_tumor, acc]
    index_ = log.index
    condition = log['id'] == id
    for i in range(len(new_data)):
        log.loc[index_[condition], eva[i]] = new_data[i]


def evaluation_pre(loss, Id, log, eva):
    # ["id", "loss"]
    loss = loss.item()
    id = Id[0]

    new_data = [id, loss]
    index_ = log.index
    condition = log['id'] == id
    for i in range(len(new_data)):
        log.loc[index_[condition], eva[i]] = new_data[i]