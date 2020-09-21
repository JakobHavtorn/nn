import numpy as np

from .module import Module


class Activation(Module):
    def update_cache(self, value):
        self.cache['value'] = value


class Sigmoid(Activation):
    def __init__(self):
        """Sigmoid"""
        super().__init__()
        self.reset_cache()

    def __str__(self): 
        return "Sigmoid()"

    def forward(self, x):
        a = 1.0 / (1 + np.exp(-x))
        self.update_cache(a)
        return a

    def backward(self, din, cache=None):
        a = self.cache['value'] if cache is None else cache
        return a * (1 - a) * din


class Tanh(Activation):
    def __init__(self):
        """Tanh"""
        super().__init__()
        self.reset_cache()

    def __str__(self):
        return "Tanh()"

    def forward(self, x):
        a = np.tanh(x)
        self.update_cache(a)
        return a

    def backward(self, din, cache=None):
        a = self.cache['value'] if cache is None else cache
        return (1 - a ** 2) * din


class ReLU(Activation):
    def __init__(self):
        """ReLU"""
        super().__init__()
        self.reset_cache()

    def __str__(self):
        return "ReLU()"

    def forward(self, x):
        a = x * (x > 0)
        self.update_cache(a)
        return a

    def backward(self, din, cache=None):
        a = self.cache['value'] if cache is None else cache
        return din * (a > 0)


class LeakyReLU(Activation):
    def __init__(self, negative_slope=0.01):
        """LeakyReLU"""
        super().__init__()
        self.negative_slope = negative_slope
        self.reset_cache()

    def __str__(self):
        return "LeakyReLU()"

    def forward(self, x):
        a1 = (x > 0) * x
        a2 = (x <= 0) * x * self.negative_slope
        self.update_cache(x)
        return a1 + a2

    def backward(self, din, cache=None):
        x = self.cache['value'] if cache is None else cache
        dout = din * (x > 0) + din * (x <= 0) * self.negative_slope
        return dout



class Softplus(Activation):
    def __init__(self):
        """Softplus"""
        super().__init__()
        self.reset_cache()

    def __str__(self):
        return "Softplus()"

    def forward(self, x):
        g = np.exp(x) + 1
        self.update_cache(g)
        return np.log(g)

    def backward(self, din, cache=None):
        g = self.cache['value'] if cache is None else cache
        return din * 1 - g ** (-1)


class Softmax(Activation):
    def __init__(self):
        """Softmax"""
        super().__init__()
        self.reset_cache()

    def __str__(self): 
        return "Softmax()"

    def forward(self, x):
        x_shifted = x - np.max(x)
        x_exp = np.exp(x_shifted)
        a = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return a

    def backward(self, din, cache=None):
        return din
