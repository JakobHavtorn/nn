# nn
## Implementation of neural network modules in numpy.
This is an implementation of some common neural network architectural modules using the Numerical Python (Numpy) library.

The overall modular structure is inspired by that of [PyTorch](https://pytorch.org/). All network modules are children of a parent ["Module"](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module). Modules include linear and convolutional layers as well as nonlinear activation functions. Entire network models are also instantiated as children of the "Module" class.

Some inspiration has also been found at the [DTU PhD Deep Learning Summer School 2015], see [website](http://deeplearningdtu.github.io/Summerschool_2015/) and [github repository](https://github.com/DeepLearningDTU/Summerschool_2015/).

Other nice sources are (https://github.com/martinkersner/cs231n/blob/master/assignment2/layers.py)

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

