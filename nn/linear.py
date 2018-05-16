import IPython
import numpy as np

from .module import Module
from .parameter import Parameter


class Linear(Module):
    """Linear module which performs an affine transformation.

    The forward transformation is
        y = x * W + b
    where the following dimensions hold true
        x:  (N, D)
        W:  (D, M)
        b:  (M
        y:  (N, M)

    Parameters
    ----------
    in_features : int
        The number of input features
    out_features : int
        The number of output features
    bias : bool, optional
        Whether or not to include a bias (the default is True)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
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

    def forward(self, x, *args):
        self.x = x
        self.a = np.dot(x, self.W.data)
        if self.b is not None:
            self.a += self.b.data
        return self.a
        
    def backward(self, dout):
        self.W.grad = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.data.T)
        if self.b is not None:
            self.b.grad = dout.sum(axis=0)
        return dx

    def update_params(self, lr):
        self.W.data = self.W.data - self.grad_W*lr
        self.b.data = self.b.data - self.grad_b*lr
