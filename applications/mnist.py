from context import nn, optim
import os
import torch
from torchvision import datasets, transforms
import IPython


dataset_name = "MNIST"
batch_size = 64

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
test_set_kwargs = {'train': False, 'download': True, 'transform': transform}

# Create dataset and DataLoader
train_set = data_set(data_dir, **train_set_kwargs)
test_set = data_set(data_dir, **test_set_kwargs)
train_loader_kwargs = {'batch_size': batch_size, 'shuffle': True}
test_loader_kwargs = {'batch_size': len(test_set), 'shuffle': True}
train_loader = torch.utils.data.DataLoader(train_set, **train_loader_kwargs)
test_loader = torch.utils.data.DataLoader(test_set, **test_loader_kwargs)

# Network
class FNNClassifier(nn.Module):
    def __init__(self, in_features, out_classes, hidden_dims=[64, 128, 64], activation=nn.ReLU, loss=nn.CrossEntropyLoss):
        IPython.embed()
        super(FNNClassifier, self).__init__()
        activations = [activation()] * len(hidden_dims) + [nn.Softmax()]
        dims = [in_features, *hidden_dims, out_classes]
        self.layers = []
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.layers.append(nn.BatchNorm1D(dims[i+1]))
            self.layers.append(activations[i])
        self.loss = loss

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, predicted_probs, targets):
        delta_out = self.loss.backward(predicted_probs, targets)
        for layer in reversed(self.layers):
            delta_out = layer.backward(delta_out)

# Train
network = FNNClassifier(28*28, 10)
