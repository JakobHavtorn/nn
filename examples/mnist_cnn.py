import IPython
import numpy as np

from context import nn, optim, utils

from loaders import get_loaders
from models import CNNClassifier


if __name__ == '__main__':
    # Model
    classifier = CNNClassifier((1, 28, 28), 10, activation=nn.ReLU, batchnorm=True, dropout=False)
    for n, m in classifier.named_modules():
        print(n, m)
    n = 0
    for p in classifier.parameters():
        n += np.prod(p.shape)
    print("Total number of parameters: " + str(n))
    # Dataset
    dataset_name = "MNIST"
    batch_size = 64
    train_loader, val_loader = get_loaders(dataset_name, batch_size)
    # Optimizer
    optimizer = optim.SGD(classifier, lr=0.001, momentum=0.9, nesterov=False, dampening=0, l1_weight_decay=0, l2_weight_decay=0)
    # Loss
    loss = nn.CrossEntropyLoss()
    # Train
    solver = utils.Solver(classifier, train_loader, val_loader, optimizer, loss)
    solver.train()
