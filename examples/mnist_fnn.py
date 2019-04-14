import os

import matplotlib.pyplot as plt
import numpy as np

from context import nn, optim, utils
from utils.constants import SAVE_DIR
from utils.utils import get_loaders


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
            if batchnorm:
                self.add_module('batchnorm_' + str(i), nn.BatchNorm1D(dims[i]))
            if dropout:
                self.add_module('dropout_' + str(i), nn.Dropout(p=dropout))
            if i > 0:
                self.add_module('activation_' + str(i), activation())
            self.add_module('linear_' + str(i), nn.Linear(dims[i], dims[i+1], bias=True))
        self.add_module('activation_' + str(i+1), nn.Softmax())

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
    classifier = FNNClassifier(28 * 28, 10, hidden_dims=[64, 32, 16], activation=nn.ReLU, batchnorm=True, dropout=False)
    classifier.summarize()
    # Dataset
    dataset_name = 'MNIST'
    batch_size = 250
    num_epochs = 10
    train_loader, val_loader = get_loaders(dataset_name, batch_size)
    # Optimizer
    optimizer = optim.SGD(classifier, lr=0.01, momentum=0.9, nesterov=False, dampening=0, l1_weight_decay=0, l2_weight_decay=0)
    # Loss
    loss = nn.CrossEntropyLoss()
    # Train
    trainer = utils.trainers.ClassificationTrainer(classifier, train_loader, val_loader, optimizer, loss, num_epochs=num_epochs, lr_decay=1.0)
    trainer.train()

    save_dir = os.path.join(SAVE_DIR, dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    val_iterations = [(epoch +1) * trainer.batches_per_epoch for epoch in range(num_epochs)]

    f, a = plt.subplots()
    a.plot(trainer.train_loss_history, '.', alpha=0.2,)
    a.plot(val_iterations, trainer.val_loss_history)
    a.set_xlabel('Iteration')
    a.set_ylabel('Negative log likelihod loss')
    a.legend(['Training', 'Validation'])
    f.savefig(save_dir + 'loss_fnn.pdf', bbox_inches='tight')
    f.savefig(save_dir + 'loss_fnn.png', bbox_inches='tight')

    f, a = plt.subplots()
    a.plot(trainer.train_acc_history, '.', alpha=0.2,)
    a.plot(val_iterations, trainer.val_acc_history)
    a.set_xlabel('Iteration')
    a.set_ylabel('Classification accuracy')
    a.legend(['Training', 'Validation'])
    f.savefig(save_dir + '/accuracy_fnn.pdf', bbox_inches='tight')
    f.savefig(save_dir + '/accuracy_fnn.png', bbox_inches='tight')
