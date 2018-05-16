# nn
## Implementation of neural network modules in numpy.
This is an implementation of some common neural network architectural modules using the Numerical Python (Numpy) library.

The overall modular structure is inspired by that of [PyTorch](https://pytorch.org/). All network modules are children of a parent `Module`(https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module). Both layers, such as linear, convolutional, and recurrent, and nonlinear activation functions are implemented as subclasses of the Module class. Network models are also instantiated as subclasses of the `Module` class and hold their layers and activation functions as attributes, effectively forming a graph.

Some inspiration has been found at the [DTU PhD Deep Learning Summer School 2015], see [website](http://deeplearningdtu.github.io/Summerschool_2015/) and [github repository](https://github.com/DeepLearningDTU/Summerschool_2015/). The batch normalization layer has been inspired by (https://github.com/martinkersner/cs231n/blob/master/assignment2/layers.py).

## Implemented modules
Currently implemented modules are
- Linear layers
    - [Linear](https://pytorch.org/docs/stable/nn.html#linear)
- Dropout layers
    - [Dropout1D](https://pytorch.org/docs/stable/nn.html#dropout)
- Normalization layers
    - [BatchNorm1D](https://pytorch.org/docs/stable/nn.html#batchnorm1d)
- Activations
    - [Sigmoid](https://pytorch.org/docs/stable/nn.html#sigmoid)
    - [Tanh](https://pytorch.org/docs/stable/nn.html#tanh)
    - [ReLU](https://pytorch.org/docs/stable/nn.html#relu)
    - [Softplus](https://pytorch.org/docs/stable/nn.html#softplus)
    - [Softmax](https://pytorch.org/docs/stable/nn.html#softmax)

## Implementation roadmap
Layers on the roadmap for implementation are
- Convolutional layers
    - [1 dimensional convolution](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d)
    - [2 dimensional convolution](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d)
    - [3 dimensional convolution](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d)
- Pooling layers
    - [1 dimensional max pooling](https://pytorch.org/docs/stable/nn.html#maxpool1d)
    - [1 dimensional average pooling](https://pytorch.org/docs/stable/nn.html#avgpool1d)
- Dropout layers
    - [2 dimensional dropout](https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout2d)
    - [3 dimensional dropout](https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout3d)
- Recurrent layers
    - [RNN](https://pytorch.org/docs/stable/nn.html#rnn)
    - [LSTM](https://pytorch.org/docs/stable/nn.html#lstm)
    - [GRU](https://pytorch.org/docs/stable/nn.html#gru)

## How to
A network model can be defined as a class. In the `__init__` method, the network should have its layers and activation functions etc. added either as either named attributes or using the `add_module` method. The latter option is well suited in cases where many similar layers are added sequentially.

Below is an example of how to construct an FNN classifier. The classifier has
- variable input and output dimensions
- variable number of hidden layers and dimensions
- specifiable activation function
- potential batchnorm and dropout layers
```python
class FNNClassifier(nn.Module):
    def __init__(self, in_features, out_classes, hidden_dims=[256, 128, 64], activation=nn.ReLU, batchnorm=False, dropout=False):
        super(FNNClassifier, self).__init__()
        dims = [in_features, *hidden_dims, out_classes]
        for i in range(len(dims)-2):
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
``` 

