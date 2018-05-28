from context import nn
import IPython


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
        for i in range(len(dims) - 2):
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


class CNNClassifier(nn.Module):
    """Convolution neural network classifier.

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

    def __init__(self, in_features, out_classes, hidden_dims=[320, 256], activation=nn.ReLU, batchnorm=False, dropout=False):
        super(CNNClassifier, self).__init__()
        # Convolutional layers
        self.add_module("Convolutional1", nn.Conv2D(1, 10, kernel_size=(5, 5)))
        # self.add_module("Batchnorm1", nn.BatchNorm2D(10))
        self.add_module("Maxpool1", nn.MaxPool2D(kernel_size=2, stride=2, padding=0))
        self.add_module("ReLU1", activation())
        # self.add_module("Convolutional2", nn.Conv2D(10, 20, kernel_size=(5, 5)))
        # self.add_module("Batchnorm2", nn.BatchNorm2D(20))
        self.add_module("Maxpool2", nn.MaxPool2D(kernel_size=2, stride=2, padding=0))
        self.add_module("ReLU2", activation())
        self.add_module("Flatten", nn.Flatten())
        # Feedforward classifier
        dims = [*hidden_dims, out_classes]
        for i in range(len(dims) - 2):
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
        for module in self._modules.values():
            x = module.forward(x)
            print(x.shape)
        return x

    def backward(self, dout):
        for module in reversed(self._modules.values()):
            dout = module.backward(dout)
