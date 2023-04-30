import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SVHNInstance(datasets.SVHN):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

def get_SVHN_dataloaders(batch_size=128, num_workers=4):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    def target_transform(target):
        return int(target) - 1

    train_loader = torch.utils.data.DataLoader(
        SVHNInstance(
            root='../datasets/SVHN/', split='train', download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        ),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root='../datasets/SVHN/', split='test', download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        ),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader