''' Food-DB Dataset class'''

from __future__ import print_function
import h5py
import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
        
        
class Food_Online(data.Dataset):
    def __init__(self, split='Training', transform=None, student_norm=None, teacher_norm=None):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.data = h5py.File('../datasets/Food_data_100.h5', 'r')
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((75750, 100, 100, 3))
        
        else:
            self.PrivateTest_data = self.data['test_data_pixel']
            self.PrivateTest_labels = self.data['test_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((25250, 100, 100, 3))

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







class Food_teacher(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = h5py.File('../datasets/Food_data_100.h5', 'r')
        self.train_data = self.data['train_data_pixel']
        self.train_labels = self.data['train_data_label']
        self.train_data = np.asarray(self.train_data)
        self.train_data = self.train_data.reshape((75750, 100, 100, 3))

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.train_data)









def get_Food_dataloader(batch_size=128, num_workers=4, is_instance=False):

    n_data = 75750
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
    ])

    transforms_teacher_train_Normalize = transforms.Normalize((0.55225873, 0.4434547, 0.33666313),
                                                              (0.26648906, 0.26966304, 0.27299368))
    transforms_student_train_Normalize = transforms.Normalize((0.5522875, 0.44350046, 0.33671987),
                                                              (0.25248525, 0.25414187, 0.2578973))
    transforms_teacher_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.5522672, 0.44518816, 0.3382316], std=[0.26212487, 0.26666662, 0.27294743])
                                                                                     (transforms.ToTensor()(crop)) for
                                                                                     crop in crops]))
    transforms_student_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.55241513, 0.4450862, 0.33795202], std=[0.24828325, 0.2513603, 0.2578531])
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

    train_loader = Food_Online(split='Training', transform=transform_train, student_norm=student_norm,
                             teacher_norm=teacher_norm)
    test_loader = Food_Online(split='PrivateTest', transform=None, student_norm=transform_test,
                            teacher_norm=transform_teacher_test)

    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size*8, shuffle=False,
                                             num_workers=num_workers)
    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader