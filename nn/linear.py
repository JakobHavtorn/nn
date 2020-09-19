import IPython
import numpy as np

from .module import Module
from .parameter import Parameter


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
        self.W = Parameter(np.zeros([out_features, in_features]))
        if bias:
            self.b = Parameter(np.zeros(out_features))
        else:
            self.b = None
        self.cache = dict(x=None)
        self.reset_parameters()

    def __str__(self):
        return "Linear({:d}, {:d}, bias={})".format(self.in_features, self.out_features, self.b is not None)

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.in_features)
        self.W.data = np.random.uniform(-stdv, stdv, self.W.shape)
        if self.b is not None:
            self.b.data = np.random.uniform(-stdv, stdv, self.b.shape)

    def forward(self, x):
        z = np.dot(x, self.W.data.T)
        if self.b is not None:
            z += self.b.data
        self.cache = dict(x=x)
        return z

    def backward(self, delta):
        x = self.cache['x']
        self.W.grad += np.dot(delta.T, x)
        dzdx = np.dot(delta, self.W.data)
        if self.b is not None:
            self.b.grad += delta.sum(axis=0)
        return dzdx


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
        self.cache = dict(x1=None, x2=None)
        self.reset_parameters()

    def __str__(self):
        return f'BiLinear(in_features_1={self.in_features_1:d}, in_features_2={self.in_features_2:d},' \
               f'out_features={self.out_features:d} bias={self.b is not None})'

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.in_features_1 + self.in_features_2)
        self.W.data = np.random.uniform(-stdv, stdv, self.W.shape)
        if self.b is not None:
            self.b.data = np.random.uniform(-stdv, stdv, self.b.shape)

    def forward(self, x1, x2=None):
        x2 = x1 if x2 is None else x2
        self.x1 = x1
        self.x2 = x2
        dzdx1 = np.einsum('oij,bj->boi', self.W.data, x2)
        z = np.einsum('bi,boi->bo', x1, dzdx1)
        if self.b is not None:
            z += self.b.data
        self.cache = dict(dzdx1=dzdx1, x1=x1, x2=None)
        return z

    def backward(self, delta):
        x1, x2, dzdx1 = self.cache['x1'], self.cache['x2'], self.cache['dzdx1']
        if x2 is None:
            x2 = x1

        print('Debuggin Bilinear')
        IPython.embed()

        self.W.grad += np.einsum('bi,bj->ij', x1, x2)
        if self.b is not None:
            self.b.grad += delta.sum(axis=0)
        if x2 is None:
            return dzdx1.sum(axis=0)
        dzdx2 = np.einsum('oij,bj->oi', self.W.data, x2)
        return dzdx1, dzdx2
        # y = x_1 W x_2 + b\\
        # \frac{dy}{dx_1} = Wx_2\\
        # \frac{dy}{dx_2} = x_1 W


