from __future__ import print_function

import torch
import numpy as np
import os
from dataset.MetaCelebA import get_CelebA_dataloader
from dataset.MetaFood import get_Food_dataloader
from dataset.MetaFairFace import get_FairFace_dataloader
from dataset.MetaPlaces_Extra69 import get_Places_Extra69_dataloader
from torch.autograd import Variable
from models.studentNet import CNN_RIS
import matplotlib.pyplot as plt
from tsne_torch import TorchTSNE as TSNE

def load_pretrained_model(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# NUM_CLASSES = 8
# _, testloader = get_CelebA_dataloader(batch_size=64, num_workers=0)

# NUM_CLASSES = 101
# _, testloader = get_Food_dataloader(batch_size=64, num_workers=0)

NUM_CLASSES = 7
_, testloader = get_FairFace_dataloader(batch_size=64, num_workers=0)

# NUM_CLASSES = 69
# _, testloader = get_Places_Extra69_dataloader(batch_size=64, num_workers=0)

def plot_features(features, labels, num_classes):
    colors = ['C' + str(i) for i in range(num_classes)]
    plt.figure(figsize=(6, 6))
    for l in range(num_classes):
        plt.scatter(
            features[labels == l, 0],
            features[labels == l, 1],
            c=colors[l], s=1, alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    if not os.path.isdir('picture_TSNE/'):
        os.mkdir('picture_TSNE/')
    plt.savefig('picture_TSNE/FairFace_dtd_FLSW.jpg', dpi=500)
    plt.show()

def feature_normalize(data):
    min_a = torch.min(data)
    max_a = torch.max(data)
    data = (data - min_a) / (max_a - min_a)
    return data

snet = CNN_RIS(num_classes=NUM_CLASSES).cuda()
snet.eval()
scheckpoint = torch.load('save/student_model/S(CNN_RIS)_T(Teacher)_FairFace_dtd_FLSW_r(0.3)_a(0.7)_b(0.0)_t(4.0)_s(320)_Re(0.0)_1/CNN_RIS_best.pth')
load_pretrained_model(snet, scheckpoint['model'])
all_features, all_labels = [], []
for batch_idx, (img, target) in enumerate(testloader):
    test_bs, ncrops, cs, hs, ws = np.shape(img)
    img = img.view(-1, cs, hs, ws)
    img = img.cuda()
    target = target.cuda()
    img, target = Variable(img), Variable(target)
    with torch.no_grad():
        _, _, _, _, features, outputs = snet(img)

        features = features.view(test_bs, ncrops, -1).mean(1)
        features = feature_normalize(features)

        all_features.append(features.data.cpu().numpy())
        all_labels.append(target.data.cpu().numpy())

all_features = np.concatenate(all_features, 0)
all_labels = np.concatenate(all_labels, 0)

# 检查所有特征是否包含 NaN 或 Inf 值
if np.isnan(all_features).any() or np.isinf(all_features).any():
    # 将 NaN 和 Inf 值替换为均值和最大值
    all_features[np.isnan(all_features)] = np.nanmean(all_features)
    all_features[np.isinf(all_features)] = np.nanmax(all_features)

    # 检查所有特征是否包含所有元素都是 NaN 或 Inf 的列
    if np.isnan(all_features).all(axis=0).any() or np.isinf(all_features).all(axis=0).any():
        # 过滤掉所有元素都是 NaN 或 Inf 的列
        all_features = all_features[:, ~np.isnan(all_features).all(axis=0)]
        all_features = all_features[:, ~np.isinf(all_features).all(axis=0)]
else:
    # 没有 NaN 或 Inf 值，无需处理
    pass

tsne = TSNE()
all_features = tsne.fit_transform(all_features)
plot_features(all_features, all_labels, NUM_CLASSES)