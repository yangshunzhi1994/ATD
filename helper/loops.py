from __future__ import print_function, division

import torch.nn.functional as F
import time
import torch
from helper.util import CosineDecay
from .util import AverageMeter, accuracy, clip_gradient
from distiller_zoo import hcl
import numpy as np

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    if opt.distill == 'ctkd':
        mlp_net = module_list[1]

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.dataset == 'CelebA' or opt.dataset == 'Food' or opt.dataset == 'Logo' or opt.dataset == 'Places_Extra69' \
                or opt.dataset == 'FairFace' or opt.dataset == 'IMDB_WIKI':
            img_teacher, img_student, target, index = data
            data_time.update(time.time() - end)
            img_teacher, img_student = img_teacher.float(), img_student.float()
            img_teacher, img_student, target, index = img_teacher.cuda(), img_student.cuda(), target.cuda(), index.cuda()
            input = img_student
        else:
            if opt.distill in ['crd']:
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                if opt.distill in ['crd']:
                    contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        if opt.dataset == 'CelebA' or opt.dataset == 'Food' or opt.dataset == 'Logo' or opt.dataset == 'Places_Extra69' \
                or opt.dataset == 'FairFace' or opt.dataset == 'IMDB_WIKI':
            _, _, _, _, _, logit_s = model_s(img_student)
            with torch.no_grad():
                _, _, _, _, _, logit_t = model_t(img_teacher)
        else:
            preact = False
            if opt.distill in ['abound']:
                preact = True
            feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                if opt.dataset == 'cifar100':
                    feat_t = [f.detach() for f in feat_t]
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'atd' or opt.distill == 'atd_at' or opt.distill == 'atd_mixup':
            loss = criterion_kd(logit_s, logit_t, target, index)
        elif opt.distill == 'ReviewKD_atd':
            loss = opt.Review * hcl(feat_s, feat_t[1:]) + criterion_kd(logit_s, logit_t, target, index)
        elif opt.distill == 'ReviewKD':
            loss_kd = hcl(feat_s, feat_t[1:])
        elif opt.distill == 'ofd':
            loss_kd = criterion_kd(feat_s[1:-1], feat_t[1:-1])
        elif opt.distill == 'dkd':
            loss_kd = criterion_kd(logit_s, logit_t, target, opt.alpha, opt.beta, opt.kd_T)
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = criterion_kd(feat_s[:-1], feat_t[:-1])
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        elif opt.distill == 'ctkd':
            mlp_net = module_list[1]
            gradient_decay = CosineDecay(max_value=opt.decay_max, min_value=opt.decay_min, num_loops=opt.decay_loops)
            decay_value = gradient_decay.get_value(epoch)

            temp = mlp_net(logit_t, logit_s, decay_value)  # (teacher_output, student_output)
            temp = opt.t_start + opt.t_end * torch.sigmoid(temp)
            temp = temp.cuda()

            loss_kd = torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(logit_s, dim=1), F.softmax(logit_t, dim=1)) * temp * temp
        elif opt.distill == 'dtd_CWSM':
            logit_s = torch.nn.functional.normalize(logit_s + 1e-10, p=2, dim=1)
            max_logit_s, _ = torch.max(logit_s, 1)
            Wx = 1/(max_logit_s + 0.00001)
            Wx = (Wx.mean() - Wx) * 40 + 10
            Wx_expand = Wx.unsqueeze(1).expand(logit_s.shape)
            KD_loss = torch.nn.KLDivLoss(reduction='none')(F.log_softmax(logit_s / Wx_expand + 1e-10, dim=1),
                                                           F.softmax(logit_t / Wx_expand + 1e-10, dim=1)).sum(1) * Wx * Wx
            loss_kd = KD_loss.mean()
        elif opt.distill == 'dtd_FLSW':
            logit_s = torch.nn.functional.normalize(logit_s + 1e-8, p=2, dim=1)
            logit_t = torch.nn.functional.normalize(logit_t + 1e-8, p=2, dim=1)
            Wx = 1 - logit_s*logit_t
            Wx = (Wx.mean() - Wx) * 40 + 10
            Wx = Wx.mean(1)
            Wx_expand = Wx.unsqueeze(1).expand(logit_s.shape)
            KD_loss = torch.nn.KLDivLoss(reduction='none')(F.log_softmax(logit_s / Wx_expand + 1e-10, dim=1),
                                                           F.softmax(logit_t / Wx_expand + 1e-10, dim=1)).sum(1) * Wx * Wx
            loss_kd = KD_loss.mean()
        else:
            raise NotImplementedError(opt.distill)

        if opt.distill == 'atd' or opt.distill == 'atd_at' or opt.distill == 'atd_mixup' or opt.distill == 'ReviewKD_atd':
            pass
        elif opt.distill == 'dkd':
            loss_cls = criterion_cls(logit_s, target)
            loss = opt.gamma * loss_cls + opt.gamma * loss_kd
        else:
            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        if opt.dataset == 'CelebA' or opt.dataset == 'Food' or opt.dataset == 'Logo' or opt.dataset == 'Places_Extra69' \
                or opt.dataset == 'FairFace' or opt.dataset == 'IMDB_WIKI':
            clip_gradient(optimizer, 0.1)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            if opt.dataset == 'CelebA' or opt.dataset == 'Food' or opt.dataset == 'Logo' or opt.dataset == 'Places_Extra69' \
                    or opt.dataset == 'FairFace' or opt.dataset == 'IMDB_WIKI':
                test_bs, ncrops, cs, hs, ws = np.shape(input)
                input = input.view(-1, cs, hs, ws)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            if opt.dataset == 'CelebA' or opt.dataset == 'Food' or opt.dataset == 'Logo' or opt.dataset == 'Places_Extra69' \
                    or opt.dataset == 'FairFace' or opt.dataset == 'IMDB_WIKI':
                _, _, _, _, _, output = model(input)
                output = output.view(test_bs, ncrops, -1).mean(1)
            else:
                # compute output
                output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, top5.avg, losses.avg