import os

import IPython
import numpy as np
import torch

from context import nn, optim, utils
from torchvision import datasets, transforms


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

# Network
class FNNClassifier(nn.Module):
    def __init__(self, in_features, out_classes, hidden_dims=[256, 128, 64], activation=nn.ReLU, batchnorm=False, dropout=False):
        super(FNNClassifier, self).__init__()
        dims = [in_features, *hidden_dims, out_classes]
        for i in range(len(dims)-2):
            self.add_module("Linear" + str(i), nn.Linear(dims[i], dims[i+1]))
            if batchnorm:
                self.add_module("BatchNorm" + str(i), nn.BatchNorm1D(dims[i+1]))
            if dropout:
                self.add_module("Dropout" + str(i), nn.Dropout(p=dropout))
            self.add_module("Activation" + str(i), activation())
        i += 1
        self.add_module("Linear" + str(i), nn.Linear(dims[i], dims[i+1]))
        self.add_module("Activation" + str(i), nn.Softmax())

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for module in self._modules.values():
            x = module.forward(x)
        return x

    def backward(self, dout):
        for module in reversed(self._modules.values()):
            dout = module.backward(dout)


classifier = FNNClassifier(28*28, 10, hidden_dims=[512, 256, 128], activation=nn.ReLU, batchnorm=True, dropout=False)
for n, m in classifier.named_modules():
    print(n, m)
n = 0
for p in classifier.parameters():
    n += np.prod(p.shape)
print("Total number of parameters: " + str(n))
dataset_name = "MNIST"
batch_size = 64
train_loader, val_loader = get_loaders(dataset_name, batch_size)
optimizer = optim.SGD(classifier, lr=0.001, momentum=0.9, nesterov=False, dampening=0, weight_decay=0)
loss = nn.CrossEntropyLoss()
solver = utils.Solver(classifier, train_loader, val_loader, optimizer, loss)
solver.train()
