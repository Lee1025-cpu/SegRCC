# _*_ coding:utf-8 _*_
# 开发人员：Lee
# 开发时间：2020-12-21 18:14
# 文件名称：Solver
# 开发工具：PyCharm
from tensorboardX import SummaryWriter
from Now_state.segmentation.evaluation import *
from torch.cuda.amp import autocast
import gc
from Now_state.segmentation.utility import *
from tqdm import tqdm
import pandas as pd
import datetime
import warnings


warnings.filterwarnings('ignore')


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, run_loader):
        self.config = config
        self.train_loader, self.valid_loader, self.run_loader = train_loader, valid_loader, run_loader
        self.train_len, self.valid_len = len(train_loader), len(valid_loader)

        self.eva = ["id", "loss", "loss_bk", "loss_kid", "loss_tu", "dc", "pc", "recall", "dc_kid", "dc_tumor", "acc"]
        self.train_log, self.valid_log = \
            pd.DataFrame(data=np.zeros([self.train_len, len(self.eva)]), columns=self.eva), \
            pd.DataFrame(data=np.zeros([self.valid_len, len(self.eva)]), columns=self.eva)

        self.train_log["id"] = self.config.k_fold_lists[self.config.train_idx]
        self.valid_log["id"] = self.config.k_fold_lists[self.config.valid_idx]

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(config.net)
        else:
            self.net = config.net
        self.lr = config.lr_init
        self.net.cuda()
        self.optimizer = optimizer(config, lr=config.lr_init)[0]
        self.lr_scheduler = lr_scheduler(config, self.optimizer)[0]

        self.writer = SummaryWriter(log_dir=config.train_fig)
        self.model_specified_path = config.model_specified_path1 if config.phase == 1 else config.model_specified_path3
        self.net_score = 0

    def train_block(self, epoch):
        self.net.train(True)

        for i, (id, image, gt) in enumerate(self.train_loader):
            with autocast():
                pred = self.net(image)
                [loss, loss_bk, loss_kidney, loss_tumor] = self.config.loss1(pred, gt)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            evaluation(pred, gt, loss, loss_bk, loss_kidney, loss_tumor, id, self.train_log, self.eva)
            gc.collect()

        self.train_log.to_excel(self.config.log_path + str(epoch) + "_t.xls")

        mean_per = self.train_log.mean()
        self.train_eva = [mean_per["loss"], mean_per["dc"], mean_per["pc"], mean_per["recall"], mean_per["dc_kid"],
                          mean_per["dc_tumor"], mean_per["loss_bk"], mean_per["loss_kid"], mean_per["loss_tu"],
                          mean_per["acc"]]
        print('[%d/%d], Loss: %.4f, dc: %4f, dc_kid: %4f, dc_tu: %4f, loss_kid: %4f, loss_tumor: %4f, prec: %.4f, '
              'rec: %.4f, acc: %.4f, loss_bk: %.4f' % (
                  epoch + 1, self.config.num_epochs, mean_per["loss"], mean_per["dc"], mean_per["dc_kid"],
                  mean_per["dc_tumor"], mean_per["loss_kid"], mean_per["loss_tu"], mean_per["pc"], mean_per["recall"],
                  mean_per["acc"], mean_per["loss_bk"]))

        self.net_path = os.path.join(self.config.model_saving, '%d_%d_%.4f' % (self.config.num_epochs, epoch,
                                                                               self.optimizer.param_groups[0]['lr']))

    def valid_block(self, epoch):
        self.net.train(False)
        self.net.eval()

        with torch.no_grad():
            for i, (id, image, gt) in enumerate(self.valid_loader):
                pred = self.net(image)
                [loss, loss_bk, loss_kidney, loss_tumor] = self.config.loss1(pred, gt)

                evaluation(pred, gt, loss, loss_bk, loss_kidney, loss_tumor, id, self.valid_log, self.eva)
                torch.cuda.empty_cache()
                gc.collect()

        self.valid_log.to_excel(self.config.log_path + str(epoch) + "_v.xls")
        mean_per = self.valid_log.mean()
        self.valid_eva = [mean_per["loss"], mean_per["dc"], mean_per["pc"], mean_per["recall"], mean_per["dc_kid"],
                          mean_per["dc_tumor"], mean_per["loss_bk"], mean_per["loss_kid"], mean_per["loss_tu"],
                          mean_per["acc"]]
        Writer(self.train_eva, self.valid_eva, epoch, self.writer, self.optimizer.param_groups[0]["lr"])

        if self.config.phase == 3:
            net_score = (self.train_eva[5] * self.train_len + self.valid_eva[5] * self.valid_len +
                         self.train_eva[4] * self.train_len + self.valid_eva[4] * self.valid_len) / \
                        (2 * (self.train_len + self.valid_len))

        elif self.config.phase == 1:
            net_score = self.train_eva[1]

        print('[%d/%d], Loss: %.4f, dc: %4f, dc_kid: %4f, dc_tu: %4f, loss_kid: %4f, loss_tumor: %4f, prec: %.4f, '
              'rec: %.4f, acc: %.4f, loss_bk: %.4f' % (
                  epoch + 1, self.config.num_epochs, mean_per["loss"], mean_per["dc"], mean_per["dc_kid"],
                  mean_per["dc_tumor"], mean_per["loss_kid"], mean_per["loss_tu"], mean_per["pc"], mean_per["recall"],
                  mean_per["acc"], mean_per["loss_bk"]))

        if net_score > self.net_score:
            self.net_score = net_score
            self.net_best = self.net.state_dict()
            print('Best net score: %.4f' % (self.net_score))
            self.new_path = self.net_path + '_' + str(net_score) + '.pkl'
            torch.save(self.net_best, self.new_path)
        print('---' * 80)

        if self.config.lr_scheduler == 'ReduceLROnPlateau':
            return self.valid_eva[0]

    def check_models(self):
        print('Running ..., Run result saving to:', self.config.run_result)
        if self.config.model_specify:
            self.net.load_state_dict(torch.load(self.model_specified_path))
            print('config.model_specify=True, model from', self.model_specified_path)
        else:
            try:
                self.net.load_state_dict(torch.load(self.new_path))
                print('config.model_specify=False, model from', self.new_path)
            except:
                raise Exception('model not found')

    def run(self):
        self.config.mode = 'run'
        self.check_models()
        print('Model loaded!')

        self.net.cuda()
        self.net.train(False)
        self.net.eval()

        with torch.no_grad():
            for i, (id, image, gt) in tqdm(enumerate(self.run_loader)):
                pred = self.net(image)
                pred = (F.softmax(pred, dim=1) > 0.5).float()
                test_saving(self.config, id, image, gt, pred)
                gc.collect()
                torch.cuda.empty_cache()

    def Loss(self, pred, gt):
        ce_part, dc_part, fc_part, lovasz_softmax_part, t_test_part = 0, 0, 0, 0, 0
        if self.config.w_c_b_k_t[0] != 0:
            ce_part += self.config.w_c_b_k_t[0] * self.loss0(pred, gt)
        if self.config.w_d_b_k_t[0] != 0:
            dc_part = self.config.w_d_b_k_t[0] * self.loss1(pred, gt)[0]
            loss_bk, loss_kidney, loss_tumor = self.config.w_d_b_k_t[0] * self.loss1(pred, gt)[1], \
                                               self.config.w_d_b_k_t[0] * self.loss1(pred, gt)[2], \
                                               self.config.w_d_b_k_t[0] * self.loss1(pred, gt)[3]
        if self.config.w_f_b_k_t[0] != 0:
            fc_part = self.config.w_f_b_k_t[0] * self.loss2(pred, gt)
        if self.config.w_l_b_k_t[0] != 0:
            lovasz_softmax_part = self.config.w_l_b_k_t[0] * self.loss3(pred, gt)
        if self.config.w_t_b_k_t[0] != 0:
            t_test_part = self.config.w_t_b_k_t[0] * self.loss4(pred, gt)

        loss = ce_part + dc_part + fc_part + lovasz_softmax_part + t_test_part

        try:
            return loss, ce_part, dc_part, fc_part, lovasz_softmax_part, loss_bk, loss_kidney, loss_tumor
        except:
            return loss, ce_part, dc_part, fc_part, lovasz_softmax_part, 0, 0, 0

    def train(self):
        if self.config.model_specify:
            self.net.load_state_dict(torch.load(self.model_specified_path))
            print('config.model_specify=True, model from', self.model_specified_path)
        for epoch in range(self.config.num_epochs):
            print('lr: %.7f' % (self.optimizer.param_groups[0]['lr']),
                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            self.train_block(epoch)
            if self.config.lr_scheduler != 'ReduceLROnPlateau':
                self.valid_block(epoch)
                self.lr_scheduler.step()

            else:
                loss_val = self.valid_block(epoch)
                self.lr_scheduler.step(loss_val)
            torch.cuda.empty_cache()


class Solver_pre(object):
    def __init__(self, config, train_loader, valid_loader, run_loader):
        self.config = config
        self.train_loader, self.valid_loader, self.run_loader = train_loader, valid_loader, run_loader
        self.train_len, self.valid_len = len(train_loader), len(valid_loader)

        self.eva = ["id", "loss"]
        self.train_log, self.valid_log = \
            pd.DataFrame(data=np.zeros([self.train_len, len(self.eva)]), columns=self.eva), \
            pd.DataFrame(data=np.zeros([self.valid_len, len(self.eva)]), columns=self.eva)

        self.train_log["id"] = self.config.k_fold_lists[self.config.train_idx]
        self.valid_log["id"] = self.config.k_fold_lists[self.config.valid_idx]

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(config.net)
        else:
            self.net = config.net
        self.lr = config.lr_init
        self.net.cuda()
        self.optimizer = optimizer(config, lr=config.lr_init)[0]
        self.lr_scheduler = lr_scheduler(config, self.optimizer)[0]

        self.writer = SummaryWriter(log_dir=config.train_fig)
        self.model_specified_path = config.model_specified_path1 if config.phase == 1 else config.model_specified_path3
        self.net_score = 0

    def train_block(self, epoch):
        self.net.train(True)

        for i, (id, image, gt) in enumerate(self.train_loader):
            with autocast():
                pred = self.net(image)
                loss = self.config.loss1(pred, gt)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            evaluation_pre(loss, id, self.train_log, self.eva)
            gc.collect()

        self.train_log.to_excel(self.config.log_path + str(epoch) + "_t.xls")

        mean_per = self.train_log.mean()
        self.train_eva = [mean_per["loss"]]
        self.net_path = os.path.join(self.config.model_saving, '%d_%d_%.4f' % (self.config.num_epochs, epoch,
                                                                               self.optimizer.param_groups[0]['lr']))

        return mean_per["loss"]

    def valid_block(self, epoch, pre_loss):
        self.net.train(False)
        self.net.eval()

        with torch.no_grad():
            for i, (id, image, gt) in enumerate(self.valid_loader):
                pred = self.net(image)
                loss = self.config.loss1(pred, gt)

                evaluation_pre(loss, id, self.valid_log, self.eva)
                torch.cuda.empty_cache()
                gc.collect()

        self.valid_log.to_excel(self.config.log_path + str(epoch) + "_v.xls")
        mean_per = self.valid_log.mean()
        self.valid_eva = [mean_per["loss"]]
        Writer_pre(self.train_eva, self.valid_eva, epoch, self.writer)

        net_score = (self.train_eva[0] ** (-1) * self.train_len + self.valid_eva[0] ** (-1) * self.valid_len) / \
                    (self.train_len + self.valid_len)

        print(
            '[%d/%d], Loss train: %.8f, valid: %.8f' % (epoch + 1, self.config.num_epochs, pre_loss, mean_per["loss"]))

        if net_score > self.net_score:
            self.net_score = net_score
            self.net_best = self.net.state_dict()
            print('Best net score: %.4f' % (self.net_score))
            self.new_path = self.net_path + '_' + str(net_score) + '.pkl'
            torch.save(self.net_best, self.new_path)
        print('---' * 30)

        if self.config.lr_scheduler == 'ReduceLROnPlateau':
            return self.valid_eva[0]

    def check_models(self):
        print('Running ..., Run result saving to:', self.config.run_result)
        if self.config.model_specify:
            self.net.load_state_dict(torch.load(self.model_specified_path))
            print('config.model_specify=True, model from', self.model_specified_path)
        else:
            try:
                self.net.load_state_dict(torch.load(self.new_path))
                print('config.model_specify=False, model from', self.new_path)
            except:
                raise Exception('model not found')

    def run(self):
        self.check_models()
        print('Model loaded!')
        self.net.cuda()
        self.net.train(False)
        self.net.eval()

        with torch.no_grad():
            for i, (id, image, gt) in tqdm(enumerate(self.run_loader)):
                pred = self.net(image)
                test_saving_pre(self.config, id, gt, pred)
                gc.collect()
                torch.cuda.empty_cache()

    def Loss(self, pred, gt):
        ce_part, dc_part, fc_part, lovasz_softmax_part, t_test_part = 0, 0, 0, 0, 0
        if self.config.w_c_b_k_t[0] != 0:
            ce_part += self.config.w_c_b_k_t[0] * self.loss0(pred, gt)
        if self.config.w_d_b_k_t[0] != 0:
            dc_part = self.config.w_d_b_k_t[0] * self.loss1(pred, gt)[0]
            loss_bk, loss_kidney, loss_tumor = self.config.w_d_b_k_t[0] * self.loss1(pred, gt)[1], \
                                               self.config.w_d_b_k_t[0] * self.loss1(pred, gt)[2], \
                                               self.config.w_d_b_k_t[0] * self.loss1(pred, gt)[3]
        if self.config.w_f_b_k_t[0] != 0:
            fc_part = self.config.w_f_b_k_t[0] * self.loss2(pred, gt)
        if self.config.w_l_b_k_t[0] != 0:
            lovasz_softmax_part = self.config.w_l_b_k_t[0] * self.loss3(pred, gt)
        if self.config.w_t_b_k_t[0] != 0:
            t_test_part = self.config.w_t_b_k_t[0] * self.loss4(pred, gt)

        loss = ce_part + dc_part + fc_part + lovasz_softmax_part + t_test_part

        try:
            return loss, ce_part, dc_part, fc_part, lovasz_softmax_part, loss_bk, loss_kidney, loss_tumor
        except:
            return loss, ce_part, dc_part, fc_part, lovasz_softmax_part, 0, 0, 0

    def train(self):
        for epoch in range(self.config.num_epochs):
            print('lr: %.7f' % (self.optimizer.param_groups[0]['lr']),
                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            pre_loss = self.train_block(epoch)
            if self.config.lr_scheduler != 'ReduceLROnPlateau':
                self.valid_block(epoch, pre_loss)
                self.lr_scheduler.step()

            else:
                loss_val = self.valid_block(epoch)
                self.lr_scheduler.step(loss_val)
            torch.cuda.empty_cache()

