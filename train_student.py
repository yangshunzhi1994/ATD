"""
the general training framework
"""

from __future__ import print_function
import numpy
import os
import argparse
import time
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
from torch.utils.data import DataLoader
from models import model_dict
from models.temp_global import Global_T
from models.imagenet import imagenet_model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.SVHN import get_SVHN_dataloaders
from dataset.CINIC10 import get_CINIC10_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.car196 import CAR196
from dataset.tinyimagenet import get_tinyimagenet_dataloader

from helper.util import adjust_learning_rate, adjust_learning_rate_wram_up

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss, OFD, DKD, ATD_Mixup
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss, ATD, Sample_entropy, build_review_kd, ATD_AT
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init

# Cross-resolution Tasks
from models.teacherNet import Teacher
from models.studentNet import CNN_RIS

from dataset.MetaCelebA import get_CelebA_dataloader
from dataset.MetaFood import get_Food_dataloader
from dataset.MetaFairFace import get_FairFace_dataloader
from dataset.MetaIMDB_WIKI import get_IMDB_WIKI_dataloader
from dataset.MetaLogo import get_Logo_dataloader
from dataset.MetaPlaces_Extra69 import get_Places_Extra69_dataloader


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
    parser.add_argument('--wram_up', type=int, default=0, help='wram up')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['cifar100', 'imagenet', 'CAR196', 'SVHN',
                                                                            'TinyImageNet', 'CINIC10', 'CelebA', 'Food',
                                                                            'Logo', 'Places_Extra69', 'FairFace', 'IMDB_WIKI'], help='dataset')
    # model
    parser.add_argument('--model_s', type=str, default='CNN_RIS',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'CNN_RIS',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'resnet18', 'MobileNetV1'])
    parser.add_argument('--path_t', type=str, default='./save/models/resnet56_vanilla/ckpt_epoch_240.pth', help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='atd', choices=['kd', 'hint', 'attention', 'similarity', 'dkd',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                        'rkd', 'pkt', 'abound', 'factor', 'nst', 'ofd',
                                                                        'atd', 'ReviewKD_atd', 'ReviewKD', 'ctkd',
                                                                       'dtd_CWSM', 'dtd_FLSW', 'atd_at', 'atd_mixup'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=1.0, help='weight balance for other losses')
    parser.add_argument('-Re', '--Review', type=float, default=0.0, help='ReviewKD_atd')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=8, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--seed', type=int, default=320, help='seed')

    # CTKD distillation
    parser.add_argument('--have_mlp', type=int, default=0)
    parser.add_argument('--mlp_name', type=str, default='global')
    parser.add_argument('--t_start', type=float, default=1)
    parser.add_argument('--t_end', type=float, default=20)
    parser.add_argument('--cosine_decay', type=int, default=1)
    parser.add_argument('--decay_max', type=float, default=1)
    parser.add_argument('--decay_min', type=float, default=0)
    parser.add_argument('--decay_loops', type=float, default=10)

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    if opt.dataset == 'cifar100':
        torch.cuda.manual_seed(opt.seed)
        torch.manual_seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        if opt.dataset == 'cifar100':
            opt.learning_rate = 0.01
        elif opt.dataset == 'TinyImageNet':
            opt.learning_rate = 0.02
        else:
            pass

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])

    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.dataset == 'CelebA' or opt.dataset == 'Food' or opt.dataset == 'Logo' or opt.dataset == 'Places_Extra69' \
            or opt.dataset == 'FairFace' or opt.dataset == 'IMDB_WIKI':
        opt.model_t = 'Teacher'
    else:
        opt.model_t = get_teacher_name(opt.path_t)
    opt.model_name = 'S({})_T({})_{}_{}_r({})_a({})_b({})_t({})_s({})_Re({})_{}'.format(opt.model_s, opt.model_t,
                    opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.kd_T, opt.seed, opt.Review, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls, dataset):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    if dataset == 'cifar100':
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
    elif dataset == 'imagenet':
        model = imagenet_model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path))
    elif dataset == 'CAR196':
        model = imagenet_model_dict[model_t](num_classes=n_cls)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
    elif dataset == 'SVHN':
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
    elif dataset == 'CINIC10':
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
    elif dataset == 'TinyImageNet':
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
    elif dataset == 'CelebA':
        model = Teacher(num_classes=n_cls).cuda()
        model.load_state_dict(torch.load('save/models/CelebA_Teacher/Best_Teacher_model.t7')['tnet'])
    elif dataset == 'Food':
        model = Teacher(num_classes=n_cls).cuda()
        model.load_state_dict(torch.load('save/models/Food_Teacher/Best_Teacher_model.t7')['tnet'])
    elif dataset == 'Logo':
        model = Teacher(num_classes=n_cls).cuda()
        model.load_state_dict(torch.load('save/models/Logo_Teacher/Best_Teacher_model.t7')['tnet'])
    elif dataset == 'Places_Extra69':
        model = Teacher(num_classes=n_cls).cuda()
        model.load_state_dict(torch.load('save/models/Places_Extra69_Teacher/Best_Teacher_model.t7')['tnet'])
    elif dataset == 'FairFace':
        model = Teacher(num_classes=n_cls).cuda()
        model.load_state_dict(torch.load('save/models/FairFace_Teacher/Best_Teacher_model.t7')['tnet'])
    elif dataset == 'IMDB_WIKI':
        model = Teacher(num_classes=n_cls).cuda()
        model.load_state_dict(torch.load('save/models/IMDB_WIKI_Teacher/Best_Teacher_model.t7')['tnet'])
    else:
        raise NotImplementedError(dataset)
    print('==> done')
    return model

def main():
    best_acc = 0
    opt = parse_option()
    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
        data = torch.randn(2, 3, 32, 32)
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloader(dataset='imagenet', batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 1000
        data = torch.randn(2, 3, 224, 224)
        model_s = imagenet_model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'CAR196':
        train_loader = DataLoader(CAR196(split='Training'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              worker_init_fn=np.random.seed(12), pin_memory=True)
        val_loader = DataLoader(CAR196(split='Testing'), batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
        n_cls = 196
        data = torch.randn(2, 3, 224, 224)
        model_s = imagenet_model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'SVHN':
        train_loader, val_loader = get_SVHN_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
        data = torch.randn(2, 3, 32, 32)
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'CINIC10':
        train_loader, val_loader = get_CINIC10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
        data = torch.randn(2, 3, 32, 32)
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'TinyImageNet':
        train_loader, val_loader = get_tinyimagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 200
        data = torch.randn(2, 3, 64, 64)
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'CelebA':
        train_loader, val_loader = get_CelebA_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 8
        model_s = CNN_RIS(num_classes=n_cls).cuda()
    elif opt.dataset == 'Food':
        train_loader, val_loader = get_Food_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 101
        model_s = CNN_RIS(num_classes=n_cls).cuda()
    elif opt.dataset == 'Logo':
        train_loader, val_loader = get_Logo_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
        model_s = CNN_RIS(num_classes=n_cls).cuda()
    elif opt.dataset == 'Places_Extra69':
        train_loader, val_loader = get_Places_Extra69_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 69
        model_s = CNN_RIS(num_classes=n_cls).cuda()
    elif opt.dataset == 'FairFace':
        train_loader, val_loader = get_FairFace_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 7
        model_s = CNN_RIS(num_classes=n_cls).cuda()
    elif opt.dataset == 'IMDB_WIKI':
        train_loader, val_loader = get_IMDB_WIKI_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 6
        model_s = CNN_RIS(num_classes=n_cls).cuda()
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.dataset)
    model_t.eval()
    model_s.eval()
    if opt.dataset == 'CelebA' or opt.dataset == 'Food' or opt.dataset == 'Logo' or opt.dataset == 'Places_Extra69' \
            or opt.dataset == 'FairFace' or opt.dataset == 'IMDB_WIKI':
        _, _, _, _, feat_t, _ = model_t(torch.randn(2, 3, 92, 92).cuda())
        _, _, _, _, feat_s, _ = model_s(torch.randn(2, 3, 44, 44).cuda())
    else:
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])
    if opt.distill == 'ReviewKD_atd' or opt.distill == 'ReviewKD':
        cnn = build_review_kd(opt.model_s, model_s, teacher=get_teacher_name(opt.path_t))
        module_list.append(cnn)
        trainable_list.append(cnn)
    else:
        module_list.append(model_s)
        trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'atd':
        temperature = Sample_entropy(opt.dataset, opt.kd_T, opt.batch_size, opt.num_workers).cuda()(model_t.cuda())
        criterion_kd = ATD(temperature, opt.gamma, opt.alpha, opt.beta)
    elif opt.distill == 'atd_at':
        temperature = Sample_entropy(opt.dataset, opt.kd_T, opt.batch_size, opt.num_workers).cuda()(model_t.cuda())
        criterion_kd = ATD_AT(temperature, opt.gamma, opt.alpha, opt.beta)
    elif opt.distill == 'atd_mixup':
        temperature = Sample_entropy(opt.dataset, opt.kd_T, opt.batch_size, opt.num_workers).cuda()(model_t.cuda())
        criterion_kd = ATD_Mixup(temperature, opt.gamma, opt.alpha, opt.beta)
    elif opt.distill == 'ReviewKD_atd':
        temperature = Sample_entropy(opt.dataset, opt.kd_T, opt.batch_size, opt.num_workers).cuda()(model_t.cuda())
        criterion_kd = ATD(temperature, opt.gamma, opt.alpha, opt.beta)
    elif opt.distill == 'ReviewKD':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'ofd':
        criterion_kd = OFD(model_t, opt.model_s, opt.batch_size)
        trainable_list.append(criterion_kd)
    elif opt.distill == 'ctkd':
        mlp = Global_T()
        criterion_kd = DistillKL(opt.kd_T)
        module_list.append(mlp)
        trainable_list.append(mlp)
    elif opt.distill == 'dtd_CWSM' or opt.distill == 'dtd_FLSW':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'dkd':
        criterion_kd = DKD()
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        criterion_kd = FSP(feat_s[:-1], feat_t[:-1])
        trainable_list.append(criterion_kd)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    module_list.cuda()
    criterion_list.cuda()
    cudnn.benchmark = True

    # routine
    for epoch in range(1, opt.epochs + 1):

        if opt.wram_up > 1:
            learning_rate = adjust_learning_rate_wram_up(epoch, opt, optimizer)
        else:
            learning_rate = adjust_learning_rate(epoch, opt, optimizer)

        f = open(opt.save_folder + '.txt', 'a')
        f.write('\n\nEpoch: %d, learning rate:  %0.8f\n' % (epoch, learning_rate))
        f.close()

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        f = open(opt.save_folder + '.txt', 'a')
        f.write("epoch:  %d, train_acc:  %0.2f, train_loss:  %0.2f, total time:  %0.2f\n" % (epoch, train_acc, train_loss, time2 - time1))
        f.write("epoch:  %d, test_acc_top1:  %0.2f, test_acc_top5:  %0.2f, test_loss:  %0.2f\n" % (epoch, test_acc, test_acc_top5, test_loss))
        f.close()

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            f = open(opt.save_folder + '.txt', 'a')
            f.write('==> Saving...\n')
            f.close()

            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            torch.save(state, save_file)

            f = open(opt.save_folder + '.txt', 'a')
            f.write('saving the best model!\n')
            f.close()

    f = open(opt.save_folder + '.txt', 'a')
    f.write('best accuracy: %0.2f\n' % best_acc)
    f.close()

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
