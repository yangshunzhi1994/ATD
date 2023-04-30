from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
from torchvision import transforms

class CAR196(data.Dataset):
    def __init__(self, split='Training'):
        self.split = split
        self.data = h5py.File('../datasets/CAR196.h5', 'r')
        normalize = transforms.Normalize(mean=[0.45170552, 0.43517223, 0.43497455],
                                         std=[0.28614414, 0.2841659, 0.2912115])
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((8144, 256, 256, 3))

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((8041, 256, 256, 3))

            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            return img, target, index

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)

        else:
            return len(self.PrivateTest_data)