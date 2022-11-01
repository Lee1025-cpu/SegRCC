import argparse
from Now_state.segmentation.utility import *
from Now_state.segmentation.image_folder import *
from Now_state.segmentation.Solver import Solver, Solver_pre
from sklearn.model_selection import KFold


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_device
    torch.set_num_threads(config.num_threads)
    torch.backends.cudnn.benchmark = True

    k_fold = KFold(n_splits=config.num_k_fold, shuffle=True)
    num_k_fold = 0
    prepare(config, num_k_fold)
    sum_para(config.net)
    # check_frozen(config.net)

    if config.mode == "pre_train":
        for train_idx, valid_idx in k_fold.split(config.k_fold_lists):
            print('num_k_fold =', num_k_fold)
            prepare(config, num_k_fold)

            config.train_idx, config.valid_idx = train_idx, valid_idx
            train_loader = get_loader_pre(mode='train', config=config)
            valid_loader = get_loader_pre(mode='valid', config=config)
            run_loader = get_loader_pre(mode='run', config=config)

            solver = Solver_pre(config, train_loader, valid_loader, run_loader)
            if not config.model_specify:
                solver.train()
                num_k_fold += 1
            else:
                solver.run()
                return

    else:
        for train_idx, valid_idx in k_fold.split(config.k_fold_lists):
            if num_k_fold < -2:
                num_k_fold += 1
                pass
            else:
                print('num_k_fold =', num_k_fold)
                prepare(config, num_k_fold)

                config.train_idx, config.valid_idx = train_idx, valid_idx
                train_loader = get_loader(mode='train', config=config)
                valid_loader = get_loader(mode='valid', config=config)
                run_loader = get_loader(mode='run', config=config)

                solver = Solver(config, train_loader, valid_loader, run_loader)
                solver.train()
                num_k_fold += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1_fix_op', default=False)
    parser.add_argument('--window', type=list, default=[-1024, -769])
    # middel processing
    parser.add_argument('--radius', type=int, default=11)
    parser.add_argument('--side', default=[10, 12, 12])

    # ReduceLROnPlateau
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--rel_threshold', type=float, default=0.2)
    parser.add_argument('--cool_down', type=int, default=2)

    root = '/data0/lyx/kits_data/'
    parser.add_argument('--cuda_device', type=str, default='2')
    parser.add_argument('--num_threads', type=int, default=8)

    parser.add_argument('--root', type=str, default=root)

    parser.add_argument('--img_aug_saving', type=str, default=root + 'auged_phase1/')  # 存放经get_item之后的case
    parser.add_argument('--img_aug_saving_phase3', type=str, default=root + 'auged_phase3_/')
    parser.add_argument('--run_path', type=str, default=root + 'test_folder/run/')

    parser.add_argument('--test_aug', type=str, default=root + 'test_aug/')
    parser.add_argument('--test_', type=str, default=root + 'test_/')
    parser.add_argument('--case_list_xls_path', default="/data0/lyx/kits_data/classification/病理_310_ccRCC_0806.xls")

    # augmentation
    parser.add_argument('--possibility', type=float, default=0.3)
    parser.add_argument('--flip_aug', type=bool, default=True)
    parser.add_argument('--rotation_aug', type=bool, default=True)
    parser.add_argument('--resize_aug', type=bool, default=True)
    parser.add_argument('--gauss_aug', type=bool, default=False)
    parser.add_argument('--gamma_aug', type=bool, default=False)

    parser.add_argument('--model_specify', default=False)
    parser.add_argument('--model_specified_path1', default='/data0/lyx/kits_data/model_saving/phase1/munet/'
                                                           'f7_d0.4A0.1_80_0.05_rre_0.0003_S_s10_d0.8_dc_mean_1201.01/'
                                                           '0/80_76_0.0000_0.9630280509591103.pkl')
    parser.add_argument('--model_specified_path3', default="/data0/lyx/kits_data/model_saving/phase3/munet/"
                                                           "f10_d0A0_30_0.0_frre_1:0.30.7_dc0.0003_0.0001Exp_s5_d0.95_l2_m_0519/"
                                                           "1/30_28_0.0001_31.106826571309817.pkl")

    parser.add_argument('--num_k_fold', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    # pre_train, train, valid, run
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--phase', type=int, default=3)

    parser.add_argument("--default_size", default=[96, 144, 224])

    # munet, DKFZ17, DeepMedic, DenseNet1, att_unet, att_unet_t, r2_unet(?), nnUnet
    parser.add_argument('--net_name', type=str, default='r2_unet')
    parser.add_argument('--feature_num', type=float, default=30)
    parser.add_argument('--feature_num_list', type=list, default=[30, 60, 120, 240, 320, 320])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--rnn_re', type=int, default=2)

    # Adam, SGD, Adagrad, RMSprop, Adadelta, Adamw, Adamax, Nadam(torch 1.6 not ava), Radam(torch 1.6 not ava), Rprop,
    # SparaseAdam, ASGD, LBFGS
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--betas', type=list, default=[0.5, 0.999])
    parser.add_argument('--etas', type=list, default=[0.5, 1.2])
    parser.add_argument('--lambd', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--step_sizes', type=list, default=[1e-6, 50])
    parser.add_argument('--t0', type=float, default=1e6)
    parser.add_argument('--weight_decay', type=float, default=0.45)
    parser.add_argument('--amsgrad', type=bool, default=False)
    parser.add_argument('--eps', type=float, default=1e-12)

    # lr
    parser.add_argument('--lr_init', type=float, default=6.73e-6)

    # loss
    parser.add_argument('--loss', type=str, default="mix")  # smooth l2
    parser.add_argument('--w_c_b_k_t', default=[0, 0.1, 0.3, 0.6])
    parser.add_argument('--w_d_b_k_t', default=[1, 0.3, 0.4, 0.3])

    parser.add_argument('--w_f_b_k_t', default=[0, 10, 15, 20])
    parser.add_argument('--w_l_b_k_t', default=[0, 10, 10, 10])
    parser.add_argument('--w_t_b_k_t', default=[0, 0, 0, 0])
    parser.add_argument('--fc_gamma', type=float, default=2)
    parser.add_argument('--average', type=str, default='m')

    # scheduler: StepLR, Cosine, Cyclic, MultiStep, Exponential, ReduceLROnPlateau, CAW, lambda
    parser.add_argument('--lr_scheduler', type=str, default='lambda')
    # StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
    parser.add_argument('--decay_gamma', type=float, default=1.1)
    parser.add_argument('--state', type=float, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-5)  # Cosine, CyclicLR, ReduceLROnPlateau, CAW
    parser.add_argument("--caw_t0", default=20)
    parser.add_argument('--caw_mul', default=2)

    parser.add_argument('--milestones', type=list, default=[10, 20, 30, 45, 60, 75, 90, 105, 115])  # MultiStepLR

    parser.add_argument('--code_state', type=str, default='0527')

    config = parser.parse_args()
    main(config)
