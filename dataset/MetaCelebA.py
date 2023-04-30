''' CelebA-DB Dataset class'''

from __future__ import print_function
import h5py
import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
        
        
class CelebA_Online(data.Dataset):
    def __init__(self, split='Training', transform=None, student_norm=None, teacher_norm=None):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.data = h5py.File('../datasets/CelebA_data_100.h5', 'r')
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((162770, 100, 100, 3))
        
        else:
            self.PrivateTest_data = self.data['test_data_pixel']
            self.PrivateTest_labels = self.data['test_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((19962, 100, 100, 3))

    def __getitem__(self, index):
        
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target, index

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
            img_student = self.student_norm(img)
        
            return img_student, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)

















class CelebA_teacher(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = h5py.File('../datasets/CelebA_data_100.h5', 'r')
        self.train_data = self.data['train_data_pixel']
        self.train_labels = self.data['train_data_label']
        self.train_data = np.asarray(self.train_data)
        self.train_data = self.train_data.reshape((162770, 100, 100, 3))

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.train_data)


















def get_CelebA_dataloader(batch_size=128, num_workers=4, is_instance=False):

    n_data = 162770
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
    ])

    transforms_teacher_train_Normalize = transforms.Normalize((0.5078246, 0.42082894, 0.37498236),
                                                              (0.30557972, 0.2825311, 0.28047305))
    transforms_student_train_Normalize = transforms.Normalize((0.5077843, 0.42079496, 0.3749455),
                                                              (0.29730356, 0.274211, 0.27247018))
    transforms_teacher_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.5014289, 0.41412634, 0.36870277], std=[0.30696684, 0.28250355, 0.28004807])
                                                                                     (transforms.ToTensor()(crop)) for
                                                                                     crop in crops]))
    transforms_student_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.5015502, 0.41397652, 0.36842453], std=[0.29862845, 0.2739866, 0.27177325])
                                                                                     (transforms.ToTensor()(crop)) for
                                                                                     crop in crops]))

    teacher_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms_teacher_train_Normalize,
    ])

    transform_teacher_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.TenCrop(92),
        transforms_teacher_test_Normalize,
    ])

    student_norm = transforms.Compose([
        transforms.Resize(44),
        transforms.ToTensor(),
        transforms_student_train_Normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(48),
        transforms.TenCrop(44),
        transforms_student_test_Normalize,
    ])

    train_loader = CelebA_Online(split='Training', transform=transform_train, student_norm=student_norm,
                             teacher_norm=teacher_norm)
    test_loader = CelebA_Online(split='PrivateTest', transform=None, student_norm=transform_test,
                            teacher_norm=transform_teacher_test)

    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size*8, shuffle=False,
                                             num_workers=num_workers)
    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader