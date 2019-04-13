import numpy as np

from .module import Module
from .parameter import Parameter


# TODO Transpose W matrix


class Linear(Module):
    """Linear module which performs an affine transformation.

    The forward transformation is
        y = x * W + b
    with the following dimensions
        x:  (N, *, I)
        W:  (I, O)
        b:  (O)
        y:  (N, *, O)
    where * means any number of additional dimensions.

    Parameters
    ----------
    in_features : int
        The number of input features (I)
    out_features : int
        The number of output features (O)
    bias : bool, optional
        Whether or not to include a bias (the default is True)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = Parameter(np.zeros([in_features, out_features]))
        if bias:
            self.b = Parameter(np.zeros(out_features))
        else:
            self.b = None
        self.reset_parameters()

    def __str__(self): 
        return "Linear({:d}, {:d}, bias={})".format(self.in_features, self.out_features, self.b is not None)

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.W.shape[1])
        self.W.data = np.random.uniform(-stdv, stdv, self.W.shape)
        if self.b is not None:
            self.b.data = np.zeros(self.b.shape)

    def forward(self, x):
        z = np.dot(x, self.W.data)
        if self.b is not None:
            z += self.b.data
        self.cache = dict(x=x)
        return z

    def backward(self, delta):
        x = self.cache['x']
        self.W.grad += np.dot(x.T, delta)
        dx = np.dot(delta, self.W.data.T)
        if self.b is not None:
            self.b.grad += delta.sum(axis=0)
        return dx


class BiLinear(Module):
    """Bilinear module which performs a bilinear transformation.

    The transformation is
        y = x_1 * W * x_2 + b
    with the following dimensions
        x_1:  (N, *, I_1)
        x_2:  (N, *, I_2)
        W:    (O, I_1, I_2)
        b:    (O)
        y:    (N, O)

    Parameters
    ----------
    in_features_1 : int
        The number of first input features (I_1)
    in_features_2 : int
        The number of second input features (I_2)
    out_features : int
        The number of output features (O)
    bias : bool, optional
        Whether or not to include a bias (the default is True)
    """
    def __init__(self, in_features_1, in_features_2, out_features, bias=True):
        super(BiLinear, self).__init__()
        self.in_features_1 = in_features_1
        self.in_features_2 = in_features_2
        self.out_features = out_features
        self.W = Parameter(np.zeros([out_features, in_features_1, in_features_2]))
        if bias:
            self.b = Parameter(np.zeros(out_features))
        else:
            self.b = None
        self.reset_parameters()

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        z = x1 @ self.W @ x2
        if self.b is not None:
            z += self.b
