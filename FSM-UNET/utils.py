# -*- coding: utf-8 -*-

import os
import math
import random
import logging
import logging.handlers
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name: str, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, f'{name}.info.log')
    info_handler = logging.handlers.TimedRotatingFileHandler(
        info_name, when='D', encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    info_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    logger.info('#----------Config info----------#')
    for k, v in config.__dict__.items():
        if not k.startswith('_'):
            logger.info(f'{k}: {v},')


def get_optimizer(config, model: nn.Module):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=config.lr, rho=config.rho, eps=config.eps, weight_decay=config.weight_decay)
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=config.lr, lr_decay=config.lr_decay, eps=config.eps, weight_decay=config.weight_decay)
    elif config.opt == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(model.parameters(), lr=config.lr, lambd=config.lambd, alpha=config.alpha, t0=config.t0, weight_decay=config.weight_decay)
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=config.lr, momentum=config.momentum, alpha=config.alpha, eps=config.eps, centered=config.centered, weight_decay=config.weight_decay)
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(model.parameters(), lr=config.lr, etas=config.etas, step_sizes=config.step_sizes)
    elif config.opt == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, dampening=config.dampening, nesterov=config.nesterov)
    else:
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.05)


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma, last_epoch=config.last_epoch)
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma, last_epoch=config.last_epoch)
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma, last_epoch=config.last_epoch)
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min, last_epoch=config.last_epoch)
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.mode, factor=config.factor, patience=config.patience,
                                                                threshold=config.threshold, threshold_mode=config.threshold_mode,
                                                                cooldown=config.cooldown, min_lr=config.min_lr, eps=config.eps)
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min, last_epoch=config.last_epoch)
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len([m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255.0 if img.max() > 1.1 else img

    msk = np.squeeze(msk)
    msk_pred = np.squeeze(msk_pred)

    if datasets != 'retinal':
        msk = (msk > 0.5).astype(np.uint8)
        msk_pred = (msk_pred > threshold).astype(np.uint8)

    plt.figure(figsize=(7, 15))
    plt.subplot(3, 1, 1); plt.imshow(img); plt.axis('off')
    plt.subplot(3, 1, 2); plt.imshow(msk, cmap='gray'); plt.axis('off')
    plt.subplot(3, 1, 3); plt.imshow(msk_pred, cmap='gray'); plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path + str(i) + '.png')
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = pred.float()
        target = target.float()
        B = pred.size(0)
        return self.bceloss(pred.reshape(B, -1), target.reshape(B, -1))


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = pred.float()
        target = target.float()
        B = pred.size(0)
        p = pred.reshape(B, -1)
        t = target.reshape(B, -1)
        inter = (p * t).sum(1)
        denom = (p.sum(1) + t.sum(1)).clamp_min(self.eps)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb, self.wd = wb, wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        return self.wd * diceloss + self.wb * bceloss


class DeepSupervisionLoss(nn.Module):
    def __init__(self, wb=1.0, wd=1.0, w_aux4=0.3, w_aux8=0.3):
        super().__init__()
        self.main = BceDiceLoss(wb=wb, wd=wd)
        self.w_aux4 = float(w_aux4)
        self.w_aux8 = float(w_aux8)

    def forward(self, out, target):
        if not isinstance(out, (tuple, list)):
            return self.main(out, target)

        seg = out[0]
        aux4 = out[1] if len(out) > 1 else None
        aux8 = out[2] if len(out) > 2 else None

        loss = self.main(seg, target)
        if (aux4 is not None) and (self.w_aux4 > 0):
            t4 = F.interpolate(target.float(), size=aux4.shape[-2:], mode='nearest')
            loss = loss + self.w_aux4 * self.main(aux4, t4)
        if (aux8 is not None) and (self.w_aux8 > 0):
            t8 = F.interpolate(target.float(), size=aux8.shape[-2:], mode='nearest')
            loss = loss + self.w_aux8 * self.main(aux8, t8)
        return loss


try:
    from thop import profile
    def cal_params_flops(model: nn.Module, size: int, logger):
        x = torch.randn(1, 3, size, size).cuda()
        flops, params = profile(model, inputs=(x,))
        print('flops', flops / 1e9)
        print('params', params / 1e6)
        total = sum(p.numel() for p in model.parameters())
        print("Total params: %.3fM" % (total / 1e6))
        logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')
except Exception:
    def cal_params_flops(model: nn.Module, size: int, logger):
        total = sum(p.numel() for p in model.parameters())
        print("Total params: %.3fM" % (total / 1e6))
        logger.info(f'THOP not available; only params counted. Total params: : {total/1e6:.4f}')
