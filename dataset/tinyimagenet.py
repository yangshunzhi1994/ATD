"""
get data loaders
"""
from __future__ import print_function

import torch
from torchvision import transforms
import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np
import torch.utils.data as data



def get_tinyimagenet_dataloader(batch_size=128, num_workers=16):

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    traindir = os.path.join('../datasets/tiny-imagenet-200/', 'train.lmdb')
    valdir = os.path.join('../datasets/tiny-imagenet-200/', 'val.lmdb')

    trainset = ImageFolderLMDB(traindir,
                               transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   normalize,
                               ]),
                                split ='Training')
    testset = ImageFolderLMDB(valdir,
                              transforms.Compose([
                                  transforms.ToTensor(),
                                  normalize,
                              ]),
                              split ='Testing')
    trainloader = torch.utils.data.DataLoader(trainset, num_workers=num_workers, batch_size=batch_size,
                                              shuffle=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, num_workers=num_workers, batch_size=batch_size,
                                             shuffle=False, pin_memory=True)

    return trainloader, testloader


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, split=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.split = split
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

        if self.split == 'Training':
            return im2arr, target, index
        else:
            return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'