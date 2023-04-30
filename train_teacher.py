from __future__ import print_function

import os
import argparse
import time
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.imagenet import imagenet_model_dict
from models import model_dict
from torch.utils.data import DataLoader
import numpy as np
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.SVHN import get_SVHN_dataloaders
from dataset.CINIC10 import get_CINIC10_dataloaders
from dataset.car196 import CAR196
from dataset.tinyimagenet import get_tinyimagenet_dataloader

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'resnet50'])
    parser.add_argument('--dataset', type=str, default='TinyImageNet', choices=['cifar100', 'CAR196', 'SVHN', 'CINIC10',
                                                                                'TinyImageNet'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}'.format(opt.model, opt.dataset)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():

    best_acc = 0
    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
        model = model_dict[opt.model](num_classes=n_cls)  # model
    elif opt.dataset == 'CAR196':
        train_loader = DataLoader(CAR196(split = 'Training'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              worker_init_fn=np.random.seed(12), pin_memory=True)
        val_loader = DataLoader(CAR196(split = 'Testing'), batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
        n_cls = 196
        model = imagenet_model_dict[opt.model](num_classes=n_cls)  # model
    elif opt.dataset == 'SVHN':
        train_loader, val_loader = get_SVHN_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
        model = model_dict[opt.model](num_classes=n_cls)  # model
    elif opt.dataset == 'CINIC10':
        train_loader, val_loader = get_CINIC10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
        model = model_dict[opt.model](num_classes=n_cls)  # model
    elif opt.dataset == 'TinyImageNet':
        train_loader, val_loader = get_tinyimagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 200
        model = model_dict[opt.model](num_classes=n_cls)
    else:
        raise NotImplementedError(opt.dataset)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    for epoch in range(1, opt.epochs + 1):
        learning_rate = adjust_learning_rate(epoch, opt, optimizer)

        f = open(opt.model_path + '/' + opt.model_name + '.txt', 'a')
        f.write('\n\nEpoch: %d, learning rate:  %0.5f\n' % (epoch, learning_rate))
        f.close()

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()

        f = open(opt.model_path + '/' + opt.model_name + '.txt', 'a')
        f.write("epoch:  %d, total time:  %0.2f, train_acc:  %0.2f, train_loss:  %0.2f\n" % (epoch, time2 - time1, train_acc, train_loss))
        f.close()

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        f = open(opt.model_path + '/' + opt.model_name + '.txt', 'a')
        f.write("epoch:  %d, test_acc_top1:  %0.2f, test_acc_top5:  %0.2f, test_loss:  %0.2f\n" % (epoch, test_acc, test_acc_top5, test_loss))
        f.close()

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            torch.save(state, save_file)

            f = open(opt.model_path + '/' + opt.model_name + '.txt', 'a')
            f.write('saving the best model!\n')
            f.close()

        # regular saving
        if epoch % opt.save_freq == 0:

            f = open(opt.model_path + '/' + opt.model_name + '.txt', 'a')
            f.write('==> Saving...\n')
            f.close()

            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    f = open(opt.model_path + '/' + opt.model_name + '.txt', 'a')
    f.write('\n\n\nbest accuracy: %0.2f', best_acc)
    f.close()

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
