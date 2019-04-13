import os

import torch

from torchvision import datasets, transforms

from utils.constants import DATA_DIR


def get_loaders(dataset_name, batch_size):
    # Dataset
    dataset = getattr(datasets, dataset_name)
    transform = transforms.Compose([transforms.ToTensor()])
    train_set_kwargs = {'train': True, 'download': True, 'transform': transform}
    val_set_kwargs = {'train': False, 'download': True, 'transform': transform}
    train_set = dataset(DATA_DIR, **train_set_kwargs)
    val_set = dataset(DATA_DIR, **val_set_kwargs)
    
    # Create dataset and DataLoader
    train_loader_kwargs = {'batch_size': batch_size, 'shuffle': True}
    val_loader_kwargs = {'batch_size': batch_size, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(train_set, **train_loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **val_loader_kwargs)
    return train_loader, val_loader
