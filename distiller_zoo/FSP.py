from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FSP(nn.Module):
    """A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning"""
    def __init__(self, s_shapes, t_shapes):
        super(FSP, self).__init__()
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        if len(s_shapes) == 4:
            self.conv0 = nn.Conv2d(16, 16, 3, 1, 1)
            self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
            self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)

    def forward(self, g_s, g_t):
        if len(g_s) == 4:
            s_fsp0 = self.conv0(g_s[0])
            s_fsp1 = self.conv1(g_s[1])
            s_fsp2 = self.conv2(g_s[2])
            s_fsp3 = self.conv3(g_s[3])
        g_s = [s_fsp0, s_fsp1, s_fsp2, s_fsp3]
        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        loss = torch.as_tensor(loss_group).sum() / len(loss_group)
        return loss

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list
