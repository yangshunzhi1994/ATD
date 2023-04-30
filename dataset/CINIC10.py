import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class CINIC10Instance(datasets.ImageFolder):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

def get_CINIC10_dataloaders(batch_size=128, num_workers=4):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    traindir = os.path.join('../datasets/CINIC10/', 'train')
    testdir = os.path.join('../datasets/CINIC10/', 'test')
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])

    trainset = CINIC10Instance(root=traindir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    testset = datasets.ImageFolder(root=testdir, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    return train_loader, test_loader