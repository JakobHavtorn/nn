import os

import IPython
import matplotlib.pyplot as plt
import numpy as np

from context import nn, optim, utils
from loaders import get_loaders


class FNNClassifier(nn.Module):
    """Feedforward neural network classifier.

    Parameters
    ----------
    in_features : int
        The number of input features.
    out_classes : int
        The number of output classes.
    hidden_dims : list
        List of the dimensions of the hidden layers. Also defines the depth of the network.
    activation : nn.Module
        The activation function to use.
    batchnorm : bool
        Whether or not to use batch normalization layers.
    dropout : bool or float
        Whether or not to include dropout and the dropout probability to use.
    """

    def __init__(self, in_features, out_classes, hidden_dims=[256, 128, 64], activation=nn.ReLU, batchnorm=False, dropout=False):
        super(FNNClassifier, self).__init__()
        dims = [in_features, *hidden_dims, out_classes]
        for i in range(len(dims) - 1):
            is_output_layer = i == len(dims) - 2
            self.add_module("linear_" + str(i), nn.Linear(dims[i], dims[i+1]))
            if batchnorm and not is_output_layer:
                self.add_module("batchnorm_" + str(i), nn.BatchNorm1D(dims[i+1]))
            if dropout and not is_output_layer:
                self.add_module("dropout_" + str(i), nn.Dropout(p=dropout))
            if not is_output_layer:
                self.add_module("activation_" + str(i), activation())
            else:
                self.add_module("activation_" + str(i), nn.Softmax())

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for module in self._modules.values():
            x = module.forward(x)
        return x

    def backward(self, dout):
        for module in reversed(self._modules.values()):
            dout = module.backward(dout)


if __name__ == '__main__':
    # Model
    classifier = FNNClassifier(28 * 28, 10, hidden_dims=[512, 256, 128], activation=nn.ReLU, batchnorm=True, dropout=False)
    classifier.summarize()
    # Dataset
    save_dir = './results/mnist/'
    dataset_name = "MNIST"
    batch_size = 250
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
    f.savefig(save_dir + 'loss_fnn.pdf', bbox_inches='tight')
    f.savefig(save_dir + 'loss_fnn.png', bbox_inches='tight')

    f, a = plt.subplots()
    a.plot(solver.train_acc_history, '.', alpha=0.2,)
    a.plot(val_iterations, solver.val_acc_history)
    a.set_xlabel('Iteration')
    a.set_ylabel('Classification accuracy')
    a.legend(['Training', 'Validation'])
    f.savefig('./results/mnist/accuracy_fnn.pdf', bbox_inches='tight')
    f.savefig('./results/mnist/accuracy_fnn.png', bbox_inches='tight')
