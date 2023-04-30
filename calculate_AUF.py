from __future__ import print_function

import torch
import numpy as np
import os
from dataset.MetaCelebA import get_CelebA_dataloader
from dataset.MetaFood import get_Food_dataloader
from dataset.MetaLogo import get_Logo_dataloader
from dataset.MetaPlaces_Extra69 import get_Places_Extra69_dataloader
from dataset.MetaFairFace import get_FairFace_dataloader
from torch.autograd import Variable
from models.studentNet import CNN_RIS

def load_pretrained_model(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# NUM_CLASSES = 8
# _, testloader = get_CelebA_dataloader(batch_size=64, num_workers=0)

# NUM_CLASSES = 101
# _, testloader = get_Food_dataloader(batch_size=64, num_workers=0)

# NUM_CLASSES = 10
# _, testloader = get_Logo_dataloader(batch_size=64, num_workers=0)

# NUM_CLASSES = 69
# _, testloader = get_Places_Extra69_dataloader(batch_size=64, num_workers=0)

NUM_CLASSES = 7
_, testloader = get_FairFace_dataloader(batch_size=64, num_workers=0)

def confusion_matrix(preds, y, NUM_CLASSES=7):
    """ Returns confusion matrix """
    assert preds.shape[0] == y.shape[0], "1 dim of predictions and labels must be equal"
    rounded_preds = torch.argmax(preds,1)
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(rounded_preds.shape[0]):
        predicted_class = rounded_preds[i]
        correct_class = y[i]
        conf_mat[correct_class][predicted_class] += 1
    return conf_mat

def ACC_evaluation(conf_mat, outputs, targets, NUM_CLASSES=None):
    conf_mat += confusion_matrix(outputs, targets, NUM_CLASSES)
    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])]) / conf_mat.sum()
    precision = [conf_mat[i, i] / (conf_mat[i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
    mAP = sum(precision) / len(precision)

    recall = [conf_mat[i, i] / (conf_mat[:, i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    F1_score = f1.mean()

    return conf_mat, acc, mAP, F1_score


snet = CNN_RIS(num_classes=NUM_CLASSES).cuda()
snet.eval()
scheckpoint = torch.load('save/student_model/S(CNN_RIS)_T(Teacher)_FairFace_dtd_FLSW_r(0.3)_a(0.7)_b(0.0)_t(4.0)_s(320)_Re(0.0)_1/CNN_RIS_best.pth')
load_pretrained_model(snet, scheckpoint['model'])
sconf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
for batch_idx, (img, target) in enumerate(testloader):
    test_bs, ncrops, cs, hs, ws = np.shape(img)
    img = img.view(-1, cs, hs, ws)
    img = img.cuda()
    target = target.cuda()
    img, target = Variable(img), Variable(target)
    with torch.no_grad():
        _, _, _, _, features, outputs = snet(img)
        outputs = outputs.view(test_bs, ncrops, -1).mean(1)
    sconf_mat, sacc, smAP, sF1_score = ACC_evaluation(sconf_mat, outputs, target, NUM_CLASSES)

print(1111111111111111111)
print(sacc)
print(smAP)
print(sF1_score)