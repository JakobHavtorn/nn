import os

import IPython
import matplotlib.pyplot as plt
import numpy as np

from context import nn, optim, utils
from loaders import get_loaders
from models import CNNClassifier


if __name__ == '__main__':
    # Model
    classifier = CNNClassifier((1, 28, 28), 10, activation=nn.ReLU, batchnorm=True, dropout=False)
    classifier.summarize()
    # Dataset
    save_dir = './results/mnist/'
    dataset_name = "MNIST"
    batch_size = 128
    num_epochs = 10
    train_loader, val_loader = get_loaders(dataset_name, batch_size)
    # Optimizer
    optimizer = optim.SGD(classifier, lr=0.01, momentum=0.9, nesterov=False, dampening=0, l1_weight_decay=0, l2_weight_decay=0)
    # Loss
    loss = nn.CrossEntropyLoss()
    # Train
    solver = utils.Solver(classifier, train_loader, val_loader, optimizer, loss, num_epochs=num_epochs, lr_decay=1.0)
    solver.train()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    val_iterations = [(epoch +1) * solver.batches_per_epoch for epoch in range(num_epochs)]

    f, a = plt.subplots()
    a.plot(solver.train_loss_history, '.', alpha=0.2,)
    a.plot(val_iterations, solver.val_loss_history)
    a.set_xlabel('Iteration')
    a.set_ylabel('Negative log likelihod loss')
    a.legend(['Training', 'Validation'])
    f.savefig(save_dir + 'loss_cnn.pdf', bbox_inches='tight')
    f.savefig(save_dir + 'loss_cnn.png', bbox_inches='tight')

    f, a = plt.subplots()
    a.plot(solver.train_acc_history, '.', alpha=0.2,)
    a.plot(val_iterations, solver.val_acc_history)
    a.set_xlabel('Iteration')
    a.set_ylabel('Classification accuracy')
    a.legend(['Training', 'Validation'])
    f.savefig('./results/mnist/accuracy_cnn.pdf', bbox_inches='tight')
    f.savefig('./results/mnist/accuracy_cnn.png', bbox_inches='tight')
