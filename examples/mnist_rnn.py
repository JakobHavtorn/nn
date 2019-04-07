import os

import IPython
import matplotlib.pyplot as plt
import numpy as np

from context import nn, optim, utils
from loaders import get_loaders
from models import RNNClassifier


class RNNClassifier(nn.Module):
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

    def __init__(self, input_size, out_classes, hidden_dims=[256], activation=nn.ReLU, batchnorm=False, dropout=False):
        super(RNNClassifier, self).__init__()

        self.add_module("RNN0", nn.RNN(input_size, hidden_dims[0], bias=True))
        # self.add_module("Linear0", nn.Linear(hidden_dims[0], hidden_dims[1], bias=True))
        self.add_module("Linear0", nn.Linear(hidden_dims[0], out_classes, bias=True))
        self.add_module("Softmax0", nn.Softmax())

        # dims = [in_features, *hidden_dims, out_classes]
        # for i in range(len(dims) - 1):
        #     is_output_layer = i == len(dims) - 2
        #     self.add_module("Linear" + str(i), nn.Linear(dims[i], dims[i+1]))
        #     if batchnorm and not is_output_layer:
        #         self.add_module("BatchNorm" + str(i), nn.BatchNorm1D(dims[i+1]))
        #     if dropout and not is_output_layer:
        #         self.add_module("Dropout" + str(i), nn.Dropout(p=dropout))
        #     if not is_output_layer:
        #         self.add_module("Activation" + str(i), activation())
        #     else:
        #         self.add_module("Activation" + str(i), nn.Softmax())

    def forward(self, x):
        # Feed each row as a vector to RNN
        assert x.shape[0] == 1, "Only supports batches of 1 example"
        x = x.reshape(x.shape[-2] ,x.shape[-1])
        x = self.RNN0.forward(x)
        x = x[-1,:].reshape(1, *x[-1,:].shape)
        x = self.Linear0.forward(x)
        x = self.Softmax0.forward(x)
        return x

    def backward(self, dout):
        for module in reversed(self._modules.values()):
            dout = module.backward(dout)


if __name__ == '__main__':
    # Model
    classifier = RNNClassifier(input_size=28, out_classes=10, )
    classifier.summarize()
    # Dataset
    save_dir = './results/mnist/'
    dataset_name = "MNIST"
    batch_size = 1
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
    f.savefig(save_dir + 'loss_rnn.pdf', bbox_inches='tight')
    f.savefig(save_dir + 'loss_rnn.png', bbox_inches='tight')

    f, a = plt.subplots()
    a.plot(solver.train_acc_history, '.', alpha=0.2,)
    a.plot(val_iterations, solver.val_acc_history)
    a.set_xlabel('Iteration')
    a.set_ylabel('Classification accuracy')
    a.legend(['Training', 'Validation'])
    f.savefig('./results/mnist/accuracy_rnn.pdf', bbox_inches='tight')
    f.savefig('./results/mnist/accuracy_rnn.png', bbox_inches='tight')
