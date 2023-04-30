import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import math

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    s_H, t_H = source.shape[2], target.shape[2]
    if s_H < t_H:
        target = F.adaptive_avg_pool2d(target, (s_H, s_H))
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

class OFD(nn.Module):
    def __init__(self, t_net, s_net_name, batch_size):
        super(OFD, self).__init__()
        self.batch_size = batch_size
        if s_net_name == 'resnet20':
            t_channels = [16,32,64]
            s_channels = [16,32,64]
        elif s_net_name == 'MobileNetV2':
            t_channels = [256, 512, 1024, 2048]
            s_channels = [12, 16, 48, 160]
        elif s_net_name == 'vgg8':
            t_channels = [256, 512, 1024, 2048]
            s_channels = [128, 256, 512, 512]
        else:
            raise NotImplementedError(s_net_name)

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

    def forward(self, s_feats, t_feats):
        feat_num = len(t_feats)
        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)
        loss = loss_distill / self.batch_size / 1000
        return loss