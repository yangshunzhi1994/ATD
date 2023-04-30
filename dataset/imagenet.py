"""
get data loaders
"""
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import six
import lmdb
import pickle
import torch.utils.data as data

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = '../datasets/Imagenet_lmdb/'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def get_imagenet_dataloader(dataset='imagenet', batch_size=128, num_workers=16):
    """
    Data Loader for imagenet
    """
    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train.lmdb')
    test_folder = os.path.join(data_folder, 'val.lmdb')

    train_set = ImageFolderLMDB(train_folder, transform=train_transform, is_train=True)
    test_set = ImageFolderLMDB(test_folder, transform=test_transform, is_train=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers//2,
                             pin_memory=True)

    return train_loader, test_loader







def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, is_train=True, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_train:
            return im2arr, target, index
        else:
            return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'