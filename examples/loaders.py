import os
from torchvision import datasets, transforms
import torch


def get_loaders(dataset_name, batch_size):
    if dataset_name == 'MNIST':
        data_set = datasets.MNIST
        mean = (0.130660742521286,)
        var = (0.30810874700546265,)
    elif dataset_name == 'FashionMNIST':
        data_set = datasets.FashionMNIST
        mean = (-2.9279166483320296e-05,)
        var = (34.189090728759766,)
    elif dataset_name == 'CIFAR10':
        data_set = datasets.CIFAR10
        mean = (0.48753172159194946, 0.47322431206703186, 0.4359692335128784)
        var = (0.24663160741329193, 0.24822315573692322, 0.2677987813949585)
    elif dataset_name == 'CIFAR100':
        data_set = datasets.CIFAR100
        mean = (0.5141649842262268, 0.47902533411979675, 0.4298681914806366)
        var = (0.2685449421405792, 0.26044416427612305, 0.28062567114830017)
    # Arguments for data set
    data_dir = os.path.join('data', dataset_name)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, var)])
    train_set_kwargs = {'train': True, 'download': True, 'transform': transform}
    val_set_kwargs = {'train': False, 'download': True, 'transform': transform}
    # Create dataset and DataLoader
    train_set = data_set(data_dir, **train_set_kwargs)
    val_set = data_set(data_dir, **val_set_kwargs)
    train_loader_kwargs = {'batch_size': batch_size, 'shuffle': True}
    val_loader_kwargs = {'batch_size': batch_size, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(train_set, **train_loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **val_loader_kwargs)
    return train_loader, val_loader
