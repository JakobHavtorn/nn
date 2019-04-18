import os

from context import nn, optim, utils, evaluators
from utils.constants import SAVE_DIR
from utils.utils import get_loaders


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

        self.input_size = input_size
        self.add_module("RNN_0", nn.RNN(input_size, hidden_dims[0], bias=True))
        # self.add_module("Linear0", nn.Linear(hidden_dims[0], hidden_dims[1], bias=True))
        self.add_module("Linear_0", nn.Linear(hidden_dims[0], out_classes, bias=True))
        self.add_module("Softmax_0", nn.Softmax())

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
        # assert x.shape[0] == 1, "Only supports batches of 1 example"
        x = x.reshape(self.input_size, x.shape[0], -1)
        x = self.RNN_0.forward(x)
        # import IPython
        # IPython.embed()
        x = self.Linear_0.forward(x)
        x = self.Softmax_0.forward(x)
        return x

    def backward(self, dout):
        for module in reversed(self._modules.values()):
            dout = module.backward(dout)


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
    classifier = RNNClassifier(28, 10, activation=nn.ReLU, batchnorm=True, dropout=False)
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
