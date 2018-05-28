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
        for i in range(len(dims) - 1):
            is_output_layer = i == len(dims) - 2
            self.add_module("Linear" + str(i), nn.Linear(dims[i], dims[i+1]))
            if batchnorm and not is_output_layer:
                self.add_module("BatchNorm" + str(i), nn.BatchNorm1D(dims[i+1]))
            if dropout and not is_output_layer:
                self.add_module("Dropout" + str(i), nn.Dropout(p=dropout))
            if not is_output_layer:
                self.add_module("Activation" + str(i), activation())
            else:
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

    def __init__(self, in_features, out_classes, feature_maps=[16, 32], hidden_dims=[512], activation=nn.ReLU, batchnorm=False, dropout=False):
        super(CNNClassifier, self).__init__()
        # Convolutional layers
        self.add_module("Convolutional0", nn.Conv2D(1, feature_maps[0], kernel_size=(5, 5)))
        # self.add_module("Batchnorm0", nn.BatchNorm2D(32))
        self.add_module("Maxpool0", nn.MaxPool2D(kernel_size=(2, 2), stride=2, padding=0))
        self.add_module("Activation0", activation())
        self.add_module("Convolutional1", nn.Conv2D(feature_maps[0], feature_maps[1], kernel_size=(5, 5)))
        # self.add_module("Batchnorm1", nn.BatchNorm2D(64))
        self.add_module("Maxpool1", nn.MaxPool2D(kernel_size=(2, 2), stride=2, padding=0))
        self.add_module("Activation1", activation())
        self.add_module("Flatten", nn.Flatten())
        # Feedforward classifier
        dims = [*hidden_dims, out_classes]
        for i in range(len(dims) - 1):
            is_output_layer = i == len(dims) - 2
            if batchnorm:
                self.add_module("BatchNorm" + str(i), nn.BatchNorm1D(dims[i]))
            self.add_module("Linear" + str(i), nn.Linear(dims[i], dims[i+1]))
            if dropout and not is_output_layer:
                self.add_module("Dropout" + str(i), nn.Dropout(p=dropout))
            if not is_output_layer:
                self.add_module("Activation" + str(i+2), activation())
            else:
                self.add_module("Activation" + str(i+2), nn.Softmax())

    def forward(self, x):
        for module in self._modules.values():
            x = module.forward(x)
        return x

    def backward(self, dout):
        for module in reversed(self._modules.values()):
            dout = module.backward(dout)
