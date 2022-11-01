# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2021-02-07 10:47
# 文件名称：utility
# 开发工具：PyCharm
import os
from Now_state.segmentation.loss import *
import numpy as np
from torch import optim
from torchsummary import summary
import SimpleITK as sitk
from torchvision import transforms as T
import xlrd
import math


def loss_config(config):
    if config.phase == 1:
        config.n_classes = 1
        if config.loss == 'dc':
            config.loss0 = BinaryDiceLoss()
        elif config.loss == 'ce':
            config.loss0 = nn.BCEWithLogitsLoss()
        elif config.loss == 'fc':
            config.loss0 = FocalLoss2d(gamma=config.fc_gamma, phase=config.phase)
        elif config.loss == 'Lovasz-Softmax':
            config.loss0 = lovasz_softmax
        elif config.loss == '0.5*ce+dc':
            config.loss0 = nn.BCEWithLogitsLoss()
            config.loss1 = BinaryDiceLoss()
        elif config.loss == '0.5*fc+dc':
            config.loss0 = FocalLoss2d(gamma=config.fc_gamma, phase=config.phase)
            config.loss1 = BinaryDiceLoss()
        else:
            raise Exception('Loss function not defined')

    elif config.phase == 2 or config.phase == 3:
        config.n_classes = 3

        config.loss0 = CELoss(weight=config.w_c_b_k_t[1:])
        config.loss1 = DiceLoss(weight=config.w_d_b_k_t[1:], phase=config.phase, average=config.average)
        config.loss2 = FocalLoss2d(weight=config.w_f_b_k_t[1:], gamma=config.fc_gamma, phase=config.phase,
                                   average=config.average)
        config.loss3 = lovasz_softmax(weight=config.w_l_b_k_t[1:])
        if config.w_t_b_k_t[0] != 0:
            config.loss4 = TestLoss(weight=config.w_t_b_k_t[1:], phase=config.phase, average=config.average)

        if config.mode == "pre_train":
            if config.loss == "l2":
                config.loss1 = nn.MSELoss()
            elif config.loss == "smooth l1":
                config.loss1 = nn.SmoothL1Loss()
            else:
                raise Exception('# phase wrong')

    else:
        raise Exception('# phase wrong')


def net_config(config):
    if config.net_name == 'munet':
        from Now_state.segmentation.models.munet import Modified3DUNet
        # Todo: there no init for modules
        config.net = Modified3DUNet(in_channels=config.batch_size, n_classes=config.n_classes,
                                    base_n_filter=config.feature_num, dropout=config.dropout)

    elif config.net_name == 'DKFZ17':
        from Now_state.segmentation.models.model_MIC_DKFZ_BraTS2017 import Unet
        config.net = Unet(c=config.batch_size, num_classes=config.n_classes, n=config.feature_num,
                          dropout=config.dropout)

    elif config.net_name == 'DeepMedic':
        from Now_state.segmentation.models.deepmedic import DeepMedic
        config.net = DeepMedic(c=config.batch_size, n1=3, n2=3, n3=4, m=12)

    elif config.net_name == 'DenseNet1':
        from Now_state.segmentation.models.densenetv1 import DenseNet
        config.net = DenseNet(n_input_channels=config.batch_size, no_max_pool=True, growth_rate=2,
                              block_config=(4, 4, 4, 4), num_init_features=4, bn_size=2,
                              drop_rate=config.dropout, num_classes=config.n_classes)

    elif config.net_name == 'att_unet':
        from Now_state.segmentation.models.unet_grid_attention_3D import AttentionUnet
        config.net = AttentionUnet(feature_scale=config.feature_num, n_classes=config.n_classes, is_deconv=True,
                                   attention_dsample=(2, 2, 2), is_batchnorm=False, dropout=config.dropout)
    elif config.net_name == 'att_unet_t':
        from Now_state.segmentation.models.unet_grid_attention_3D import AttentionUnet_t
        config.net = AttentionUnet_t(feature_scale=config.feature_num, n_classes=config.n_classes, is_deconv=True,
                                     attention_dsample=(2, 2, 2), is_batchnorm=False, dropout=config.dropout)

    elif config.net_name == 'r2_unet':
        from Now_state.segmentation.models.r2unet import R2U_Net
        config.net = R2U_Net(img_ch=config.batch_size, output_ch=config.n_classes, t=config.rnn_re,
                             feature_scale=config.feature_num, dropout=config.dropout)

    elif config.net_name == "nnUnet":
        from Now_state.segmentation.models.nnunet_test import nnUnet
        config.net = nnUnet(feature_num=config.feature_num_list, c=config.batch_size, num_classes=config.n_classes,
                            dropout=config.dropout, max_op="conv", up_sample="convt", pre_activation=True)


def prepare(config, num_k_fold):
    loss_config(config)
    net_config(config)

    aug_type = ''
    if config.flip_aug:
        aug_type += 'f'
    if config.rotation_aug:
        aug_type += 'r'
    if config.resize_aug:
        aug_type += "re"
    if config.gauss_aug:
        aug_type += 'G'
    if config.gamma_aug:
        aug_type += 'g'

    Optimizer, op = optimizer(config, config.lr_init)[0], optimizer(config, config.lr_init)[1]
    lr_sc = lr_scheduler(config, Optimizer)[1]

    name = 'f' + str(config.feature_num) + '_d' + str(config.dropout) + op + str(config.weight_decay) + '_' + \
           str(config.num_epochs) + '_' + str(config.possibility) + '_' + aug_type + '_'

    if config.phase == 3:
        if config.w_c_b_k_t[0] != 0:
            name += str(config.w_c_b_k_t[0]) + ':' + str(config.w_c_b_k_t[1]) + str(config.w_c_b_k_t[2]) + \
                    str(config.w_c_b_k_t[3]) + '_ce'

        if config.w_d_b_k_t[0] != 0:
            name += str(config.w_d_b_k_t[0]) + ':' + str(config.w_d_b_k_t[1]) + str(config.w_d_b_k_t[2]) + \
                    str(config.w_d_b_k_t[3]) + '_dc'

        if config.w_f_b_k_t[0] != 0:
            name += str(config.w_f_b_k_t[0]) + ':' + str(config.w_f_b_k_t[1]) + str(config.w_f_b_k_t[2]) + \
                    str(config.w_f_b_k_t[3]) + '_fc'

        if config.w_l_b_k_t[0] != 0:
            name += str(config.w_l_b_k_t[0]) + ':' + str(config.w_l_b_k_t[1]) + str(config.w_l_b_k_t[2]) + \
                    str(config.w_l_b_k_t[3]) + '_lova'

        if config.w_t_b_k_t[0] != 0:
            name += str(config.w_t_b_k_t[0]) + ':' + str(config.w_t_b_k_t[1]) + str(config.w_t_b_k_t[2]) + \
                    str(config.w_t_b_k_t[3]) + '_T'

    name += str(config.lr_init) + '_' + str(config.min_lr) + lr_sc + '_s' + str(config.state) + '_d' + \
            str(config.decay_gamma) + '_' + config.loss + '_' + config.average + '_' + config.code_state + '/'

    config.train_fig = config.root + 'criterion_fig/phase' + str(config.phase) + '/' + config.net_name + '/' + name + \
                       '/' + str(num_k_fold) + '/'
    config.model_saving = config.root + 'model_saving/phase' + str(config.phase) + '/' + config.net_name + '/' + name + \
                          '/' + str(num_k_fold) + '/'
    config.test_result = config.root + 'test_result/phase' + str(config.phase) + '/' + config.net_name + '/' + name + \
                         '/' + str(num_k_fold) + '/'
    config.run_result = config.root + 'run_result/phase' + str(config.phase) + '/' + config.net_name + '/' + name + '/'
    config.log_path = config.root + 'performance_log/' + config.net_name + '/' + name + '/' + str(num_k_fold) + '/'

    if not os.path.exists(config.train_fig):
        os.makedirs(config.train_fig)
    if not os.path.exists(config.model_saving):
        os.makedirs(config.model_saving)
    if not os.path.exists(config.test_result):
        os.makedirs(config.test_result)
    if not os.path.exists(config.run_result):
        os.makedirs(config.run_result)
    if not os.path.exists(config.test_aug):
        os.makedirs(config.test_aug)
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    config.k_fold_lists = []
    if config.mode != "pre_train":
        for file in os.listdir(config.img_aug_saving):
            if 'case' in file and '_lab' not in file:
                config.k_fold_lists.append(file.split('_')[0] + "_" + file.split("_")[1])
    else:
        data = xlrd.open_workbook(config.case_list_xls_path).sheet_by_index(1)
        k_fold_lists = data.col_values(1)[1:]
        for file in k_fold_lists:
            config.k_fold_lists.append(file + "_0_img.nii.gz")

    config.k_fold_lists = np.array(config.k_fold_lists)


def sum_para(net):
    num_params = 0
    for p in net.parameters():
        num_params += p.numel()

    print("The number of parameters: {}".format(num_params))


def check_frozen(net):
    for k, v in net.named_parameters():
        if k != 'xxx.weight' and k != 'xxx.bias':
            print(v.requires_grad)  # 理想状态下，所有值都是False

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)


def summary_net(net):
    print('For input_size=(1, 128, 128, 128):')
    model = net.cuda()
    summary(model, input_size=(1, 128, 128, 128))


def optimizer(config, lr):
    eps = config.eps
    betas = (config.betas[0], config.betas[1])
    etas = (config.etas[0], config.etas[1])
    step_sizes = (config.step_sizes[0], config.step_sizes[1])
    weight_decay = config.weight_decay
    amsgrad = config.amsgrad
    lambd = config.lambd
    alpha = config.alpha
    t0 = config.t0

    if config.optimizer == 'Adam':
        # optimizer = optim.Adam(list(config.net.parameters()), lr, betas,
        #                        weight_decay=weight_decay, eps=eps, amsgrad=amsgrad)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, config.net.parameters()), lr, betas,
                               weight_decay=weight_decay, eps=eps, amsgrad=amsgrad)
        op = 'A'
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(list(config.net.parameters()), lr, momentum=0.8, weight_decay=weight_decay, nesterov=True)
        op = 'S'
    elif config.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(list(config.net.parameters()), lr, lr_decay=0.3, weight_decay=weight_decay)
        op = 'Ag'
    elif config.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(list(config.net.parameters()), lr, momentum=0.9, weight_decay=weight_decay)
        op = 'R'
    elif config.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(list(config.net.parameters()), lr, eps=eps, weight_decay=weight_decay)
        op = 'Ad'
    elif config.optimizer == 'Adamw':
        optimizer = optim.AdamW(list(config.net.parameters()), lr, betas, eps=eps, weight_decay=weight_decay,
                                amsgrad=amsgrad)
        op = 'Aw'
    elif config.optimizer == 'Adamax':
        optimizer = optim.Adamax(list(config.net.parameters()), lr, betas, eps, weight_decay=weight_decay)
        op = 'Ax'
    elif config.optimizer == 'Nadam':
        optimizer = optim.NAdam(list(config.net.parameters()), lr, betas, eps, weight_decay=weight_decay,
                                momentum_deacy=config.momentum_deacy)
        op = 'Na'
    elif config.optimizer == 'Radam':
        optimizer = optim.RAdam(list(config.net.parameters()), lr, betas, eps, weight_decay=weight_decay)
        op = 'Ra'
    elif config.optimizer == 'Rprop':
        optimizer = optim.Rprop(list(config.net.parameters()), lr, etas=etas, step_sizes=step_sizes)
        op = 'Rp'
    elif config.optimizer == 'SparseAdam':
        optimizer = optim.SparseAdam(list(config.net.parameters()), lr, betas=betas, eps=eps)
        op = 'SA'
    elif config.optimizer == 'ASGD':
        optimizer = optim.ASGD(list(config.net.parameters()), lr, lambd=lambd, alpha=alpha, weight_decay=weight_decay)
        op = 'AS'
    elif config.optimizer == 'LBFGS':
        # optimizer = optim.LBFGS(list(config.net.parameters()), lr, max_iter=)
        op = 'L'
    return [optimizer, op]


def nre_lr(x):
    # 100版本
    # if x < 45:
    #     return (((1 + math.cos(x * math.pi / 15)) / 2) ** 1.0) * 0.38 + 0.62
    # elif 45 <= x < 50:
    #     return 0.62
    # elif 50 <= x < 95:
    #     return (((1 + math.cos((x - 50) * math.pi / 15)) / 2) ** 1.0) * 1.99 + 2.60
    # else:
    #     return 2.6

    # 70版本
    # if x < 15:
    #     return (((1 + math.cos(x * math.pi / 15)) / 2) ** 1.0) * 0.38 + 0.62
    # elif 15 <= x < 60:
    #     return (((1 + math.cos((x - 15) * math.pi / 15)) / 2) ** 1.0) * 1.99 + 2.60
    # elif 60 <= x:
    #     return 4.6

    # new 100版本
    # if x < 35:
    #     return (((1 + math.cos(x * math.pi / 15)) / 2) ** 1.0) * 0.38 + 0.62
    # elif 35 <= x < 80:
    #     return (((1 + math.cos((x - 35) * math.pi / 15)) / 2) ** 1.0) * 1.99 + 2.60
    # elif 80 <= x:
    #     return (((1 + math.cos((x - 80) * math.pi / 20)) / 2) ** 1.0) * 1.98 + 0.62

    # 80版本
    return (((1 + math.cos(x * math.pi / 15)) / 2) ** 1.0) * 0.38 + 0.62


def lr_scheduler(config, optimizer):
    if config.lr_scheduler == 'Step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.num_epochs / config.state,
                                                 gamma=config.decay_gamma)
        lr_sc = 'S'
    elif config.lr_scheduler == 'Cosine':
        # cycle =  2 * T_max
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config.num_epochs / config.state),
                                                            eta_min=config.min_lr, last_epoch=-1)
        lr_sc = 'Cos'
    elif config.lr_scheduler == 'Cyclic':
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.min_lr, max_lr=config.lr_init)
        lr_sc = 'Cy'
    elif config.lr_scheduler == 'MultiStep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.decay_gamma)
        lr_sc = 'MS'
    elif config.lr_scheduler == 'Exponential':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_gamma)
        lr_sc = 'Exp'
    elif config.lr_scheduler == 'ReduceLROnPlateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.decay_gamma,
                                                            patience=config.patience, threshold_mode='rel',
                                                            threshold=config.rel_threshold, cooldown=config.cool_down,
                                                            min_lr=config.min_lr)
        lr_sc = 'R'
    elif config.lr_scheduler == "CAW":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.caw_t0,
                                                                      T_mult=config.caw_mul, eta_min=config.min_lr)
        lr_sc = 'CAW'
    elif config.lr_scheduler == "lambda":
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, nre_lr, last_epoch=-1)
        lr_sc = 'lamd'

    return [lr_scheduler, lr_sc]


def Writer(train_criterion, valid_criterion, epoch, writer, lr):
    writer.add_scalars('loss', {'train_loss': train_criterion[0], 'valid_loss': valid_criterion[0],
                                'train_loss_bk': train_criterion[6], 'valid_loss_bk': valid_criterion[6],
                                'train_loss_kid': train_criterion[7], 'valid_loss_kid': valid_criterion[7],
                                'train_loss_tumor': train_criterion[8], 'valid_loss_tumor': valid_criterion[8],
                                }, epoch)
    writer.add_scalars('loss_lr_log10', {'train_loss': train_criterion[0], 'valid_loss': valid_criterion[0],
                                         'train_loss_bk': train_criterion[6], 'valid_loss_bk': valid_criterion[6],
                                         'train_loss_kid': train_criterion[7], 'valid_loss_kid': valid_criterion[7],
                                         'train_loss_tumor': train_criterion[8], 'valid_loss_tumor': valid_criterion[8],
                                         }, np.log10(lr))
    writer.add_scalars('loss_lr_log2', {'train_loss': train_criterion[0], 'valid_loss': valid_criterion[0],
                                        'train_loss_bk': train_criterion[6], 'valid_loss_bk': valid_criterion[6],
                                        'train_loss_kid': train_criterion[7], 'valid_loss_kid': valid_criterion[7],
                                        'train_loss_tumor': train_criterion[8], 'valid_loss_tumor': valid_criterion[8],
                                        }, np.log2(lr))

    writer.add_scalars('dice', {'train_dice': train_criterion[1], 'valid_dice': valid_criterion[1],
                                'train_dice_kid': train_criterion[4], 'valid_dice_kid': valid_criterion[4],
                                'train_dice_tumor': train_criterion[5], 'valid_dice_tumor': valid_criterion[5],
                                }, epoch)
    writer.add_scalars('precision', {'train_precision': train_criterion[2], 'valid_precision': valid_criterion[2]
                                     }, epoch)
    writer.add_scalars('recall', {'train_recall': train_criterion[3], 'valid_recall': valid_criterion[3]}, epoch)
    writer.add_scalars('acc', {'train_acc': train_criterion[9], 'valid_acc': valid_criterion[9]}, epoch)


def Writer_pre(train_criterion, valid_criterion, epoch, writer):
    writer.add_scalars('loss', {'train_loss': train_criterion[0], 'valid_loss': valid_criterion[0]}, epoch)


def test_saving(config, case_name, image, gt, pred):
    if config.mode == 'run':
        saving_root = config.run_result
    elif config.mode == 'test':
        saving_root = config.test_result

    if not os.path.exists(saving_root + case_name[0] + '_image.nii.gz'):
        image = image.cpu().numpy().squeeze(0).squeeze(0).astype(np.float32).swapaxes(1, 2)
        image = sitk.GetImageFromArray(image)
        sitk.WriteImage(image, saving_root + case_name[0] + '_image.nii.gz')

        if config.phase == 1:
            try:
                gt = gt.cpu().numpy().squeeze(0).squeeze(0).astype(np.int).swapaxes(1, 2)
            except:
                pass
            pred = pred.cpu().numpy().squeeze(0).squeeze(0).astype(np.int).swapaxes(1, 2)
        elif config.phase == 2 or config.phase == 3:
            pred = torch.argmax(pred.squeeze(0), dim=0).cpu().numpy().astype(np.int).swapaxes(1, 2)

            if config.mode != 'run':
                gt = torch.argmax(gt.squeeze(0), dim=0).cpu().numpy().astype(np.int).swapaxes(1, 2)

        if config.mode == 'run':
            try:
                gt = sitk.GetImageFromArray(gt)
                sitk.WriteImage(gt, saving_root + case_name[0] + '_gt.nii.gz')
            except:
                pass

        pred = sitk.GetImageFromArray(pred)
        sitk.WriteImage(pred, saving_root + case_name[0] + '_pred.nii.gz')


def test_saving_pre(config, case_name, gt, pred):
    saving_root = config.run_result

    gt = gt.cpu().numpy().squeeze(0).squeeze(0).astype(np.float32).swapaxes(1, 2)
    gt = sitk.GetImageFromArray(gt)
    sitk.WriteImage(gt, saving_root + case_name[0] + '_image.nii.gz')

    pred0 = pred[0][0].cpu().numpy().astype(np.float32).swapaxes(1, 2)
    pred0 = sitk.GetImageFromArray(pred0)
    sitk.WriteImage(pred0, saving_root + case_name[0] + '_pred0.nii.gz')

    pred1 = pred[0][1].cpu().numpy().astype(np.float32).swapaxes(1, 2)
    pred1 = sitk.GetImageFromArray(pred1)
    sitk.WriteImage(pred1, saving_root + case_name[0] + '_pred1.nii.gz')


def Morphological_dilate(pred, radius):
    """Morphological dilate to pred"""
    binary_dilate_filter = sitk.BinaryDilateImageFilter()
    binary_dilate_filter.SetKernelRadius(radius)

    binary_dilate_filter.SetForegroundValue(1)
    # binary_dilate_filter.SetBackgroundValue(0)

    top_segmentation = binary_dilate_filter.Execute(sitk.GetImageFromArray(pred))

    pred = sitk.GetArrayFromImage(top_segmentation)

    return pred


def boundingbox(img, pred, side):
    """Bounding box for prediction"""
    box = np.where(pred)
    c_min, c_max = np.min(box[0]), np.max(box[0])
    h_min, h_max = np.min(box[1]), np.max(box[1])
    w_min, w_max = np.min(box[2]), np.max(box[2])

    if abs(c_min) > side[0] and abs(c_max - pred.shape[0]) > side[0]:
        c_min -= side[0]
        c_max += side[0]

    if abs(h_min) > side[1] and abs(h_max - pred.shape[1]) > side[1]:
        h_min -= side[1]
        h_max += side[1]

    if abs(w_min) > side[2] and abs(w_max - pred.shape[2]) > side[2]:
        w_min -= side[2]
        w_max += side[2]

    pred = pred[c_min: c_max, h_min: h_max, w_min: w_max]
    img = img[c_min: c_max, h_min: h_max, w_min: w_max]

    return img, pred


def padding(img, pred):
    padding_num = 32
    shape_x = pred.shape
    pad_c, pad_h, pad_w = padding_num - shape_x[0] % padding_num, padding_num - shape_x[1] % padding_num, \
                          padding_num - shape_x[2] % padding_num
    pad_c = [int(pad_c / 2), pad_c - int(pad_c / 2)]
    pad_h = [int(pad_h / 2), pad_h - int(pad_h / 2)]
    pad_w = [int(pad_w / 2), pad_w - int(pad_w / 2)]

    pred = np.pad(pred, ((pad_c[0], pad_c[1]), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), mode='constant',
                  constant_values=0)
    img = np.pad(img, ((pad_c[0], pad_c[1]), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), mode='constant',
                 constant_values=0)

    return img, pred


def arr_to_tensor(image):
    """Transform arr to tensor, float32, gpu, require_grad(True), one_hot_encoder(if phase2) """
    Transform = []
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    image = image.astype(np.float32)
    image = image.transpose((1, 2, 0))
    image = Transform(image)

    image = image.unsqueeze(0).unsqueeze(0).cuda()
    image = image / image.max()
    image.requires_grad = True

    return image


def middel_processing(image, pred, case_name, config):
    image = image.cpu().numpy().squeeze(0).squeeze(0).astype(np.float32)
    pred = torch.argmax(pred.squeeze(0), dim=0).cpu().numpy().astype(np.int)

    pred = Morphological_dilate(pred, radius=config.radius)
    assert image.shape == pred.shape, 'shape wrong'

    sum_gt1 = sum(sum(sum(pred)))

    image, pred = boundingbox(image, pred, side=config.side)
    sum_gt2 = sum(sum(sum(pred)))

    image, pred = padding(image, pred)
    sum_gt3 = sum(sum(sum(pred)))

    if not (sum_gt1 == sum_gt2 == sum_gt3):
        print(case_name, sum_gt1 - sum_gt3)

    return arr_to_tensor(image)


def Morphological_dilate(pred, radius):
    """Morphological dilate to pred"""
    # TODO: check the d h w or d w h
    binary_dilate_filter = sitk.BinaryDilateImageFilter()
    binary_dilate_filter.SetKernelRadius(radius)

    binary_dilate_filter.SetForegroundValue(1)
    seg = binary_dilate_filter.Execute(pred)

    return seg


def net_fix(pred, pred_phase1, case_name):
    pred_phase1 = pred_phase1.permute(0, 1, 3, 2)
    # assert pred_phase1.shape[1:] == pred.shape[2:], 'shape wrong with ' + case_name[0] + ' ' +  \
    #                                                 str(pred_phase1.shape[1:]) + ' ' + str(pred.shape[2:])
    phase1_reslut = torch.zeros(pred.shape).cuda()
    phase1_reslut[0, 1][pred_phase1[0] == 1] = 1
    # # 先框住
    # pred[0, 1] = pred[0, 1] * phase1_reslut[0, 1]
    # pred[0, 2] = pred[0, 2] * phase1_reslut[0, 1]
    # plan 1:在phase1预测的mask下对phase3未填充部分直接填充label1（或2）
    # TODO: Now, just change pred 2 with phase1_reslut[1] - pred[1]
    # pred[0, 1] = phase1_reslut[0, 1] - pred[0, 2]
    #
    # return pred

    # plan 2:对phase3做膨胀，之后用phase1的mask去框
    Pred = pred.clone()
    Pred[0, 2][Pred[0, 2] == 1] = 2
    # 膨胀之后的效果很差
    Pred = sitk.GetImageFromArray((Pred[0, 1] + Pred[0, 2]).cpu().numpy().astype(np.int))
    sitk.WriteImage(Pred, ('/data0/lyx/kits_data/test_result/phase3/DKFZ17/'
                           'f10_d0.3A0.1_120_0.05_rre_0.0003_S_s12_d0.83_dc_mean_0610_t_plan2_no_box/test_b.nii.gz'))

    sitk.WriteImage(Morphological_dilate(Pred, 1), ('/data0/lyx/kits_data/test_result/phase3/DKFZ17/'
                                                    'f10_d0.3A0.1_120_0.05_rre_0.0003_S_s12_d0.83_dc_mean_0610_t_plan2_no_box/test_m.nii.gz'))

    seg = sitk.GetArrayFromImage(Morphological_dilate(Pred, 10))
    Seg = torch.zeros(pred.shape).cuda()
    Seg[0, 0][seg == 0] = 1
    Seg[0, 1][seg == 1] = 1
    Seg[0, 2][seg == 2] = 1

    Seg[0, 1] = Seg[0, 1] * phase1_reslut[0, 1]
    Seg[0, 2] = Seg[0, 2] * phase1_reslut[0, 1]

    return Seg
