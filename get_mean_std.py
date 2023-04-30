from torch.utils.data import DataLoader
import numpy as np
from dataset.car196 import CAR196

train_set = CAR196(split='Training')
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0, worker_init_fn=np.random.seed(12))

train_mean=0
train_std=0
epoch_mean=0
epoch_std=0

if __name__ == '__main__':
    for epoch in range(1, 10):
        for batch_idx, (inputs, _, _) in enumerate(train_loader):
            train_mean += np.mean(inputs.numpy(), axis=(0,2,3))
            train_std += np.std(inputs.numpy(), axis=(0,2,3))
            mean = train_mean/(batch_idx+1)
            std = train_std/(batch_idx+1)
        train_mean=0
        train_std=0
        epoch_mean += mean
        epoch_std += std
    print('------train--------')
    print (epoch_mean/epoch, epoch_std/epoch)