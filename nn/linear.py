import numpy as np
from .module import Module
from .parameter import Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = Parameter(np.zeros(in_features, out_features))
        if bias:
            self.b = Parameter(np.zeros(out_features))
        else:
            self.b = None
        self.reset_parameters()

    def __str__(self): 
        return "Linear({d}, {d})".format(self.in_features, self.out_features)

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.W.data.shape[1])
        self.W.data = np.random.uniform(-stdv, stdv, self.W.data.shape)
        if self.b is not None:
            self.b.data = np.zeros(self.b.data.shape)

    def forward(self, x, *args):
        self.x = x
        self.a = np.dot(x, self.W.data) + self.b.data
        return self.a
        
    def backward(self, dout):
        self.W.grad = np.dot(self.x.T, dout)
        if self.b is not None:
            self.b.grad = dout.sum(axis=0)
        din = np.dot(dout, self.W.data.T)
        return din

    def update_params(self, lr):
        self.W.data = self.W.data - self.grad_W*lr
        self.b.data = self.b.data - self.grad_b*lr
