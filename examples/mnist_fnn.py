import os

from context import nn, optim, utils, evaluators
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

    def backward(self, delta):
        for module in reversed(self._modules.values()):
            delta = module.backward(delta)


if __name__ == '__main__':
    # Dataset
    dataset_name = 'MNIST'
    batch_size = 250
    max_epochs = 20
    max_epochs_no_improvement = 10
    train_loader, val_loader = get_loaders(dataset_name, batch_size)

    # Checkpoint dir
    checkpoint_dir = os.path.join(SAVE_DIR, dataset_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Model
    classifier = FNNClassifier(28 * 28, 10, hidden_dims=[128, 64, 32, 16], activation=nn.ReLU, batchnorm=True, dropout=0.2)
    classifier.summarize()

    # Optimizer
    optimizer = optim.Adam(classifier.parameters, lr=0.001, l1_weight_decay=0, l2_weight_decay=0)
    # optimizer = optim.SGD(classifier.parameters, lr=0.001, momentum=0.9, nesterov=True, l1_weight_decay=0, l2_weight_decay=0)

    # Loss
    loss = nn.CrossEntropyLoss()

    # Learning rate schedule
    lr_scheduler = None  # optim.CosineAnnealingLR(optimizer, T_max=5, decay_eta_max_half_time=1)

    # Evaluators
    train_evaluator = evaluators.MulticlassEvaluator(n_classes=10)
    val_evaluator = evaluators.MulticlassEvaluator(n_classes=10)

    # Trainer
    trainer = utils.trainers.ClassificationTrainer(classifier, optimizer, loss, train_loader, val_loader,
                                                   train_evaluator, val_evaluator,
                                                   lr_scheduler=lr_scheduler, max_epochs=max_epochs, 
                                                   max_epochs_no_improvement=max_epochs_no_improvement, 
                                                   checkpoint_dir=checkpoint_dir)
    trainer.train()
