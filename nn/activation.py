import numpy as np

from .module import Module


class Activation(Module):
    def update_cache(self, value):
        self.cache['value'] = value


class Sigmoid(Activation):
    def __init__(self):
        """Sigmoid
        """
        super().__init__()
        self.reset_cache()

    def __str__(self): 
        return "Sigmoid()"
    
    def forward(self, x):
        a = 1.0 / (1 + np.exp(-x))
        self.update_cache(a)
        return a
        
    def backward(self, din):
        a = self.cache['value']
        return a * (1 - a) * din


class Tanh(Activation):
    def __init__(self):
        """Tanh
        """
        super().__init__()
        self.reset_cache()

    def __str__(self):
        return "Tanh()"

    def forward(self, x):
        a = np.tanh(x)
        self.update_cache(a)
        return a

    def backward(self, din):
        a = self.cache['value']
        return (1 - a ** 2) * din


class ReLU(Activation):
    def __init__(self):
        """ReLU
        """
        super().__init__()
        self.reset_cache()

    def __str__(self):
        return "ReLU()"

    def forward(self, x):
        a = np.maximum(0, x)
        self.update_cache(a)
        return a

    def backward(self, din):
        a = self.cache['value']
        return din * (a > 0).astype(a.dtype)
        

class Softplus(Activation):
    def __init__(self):
        """Softplus
        """
        super().__init__()
        self.reset_cache()

    def __str__(self):
        return "Softplus()"

    def forward(self, x):
        g = np.exp(x) + 1
        self.update_cache(g)
        return np.log(g)

    def backward(self, din):
        g = self.cache['value']
        return din * 1 - g ** (-1)


class Softmax(Activation):
    def __init__(self):
        """Softmax
        """
        super().__init__()
        self.reset_cache()

    def __str__(self): 
        return "Softmax()"
    
    def forward(self, x):
        x_shifted = x - np.max(x)
        x_exp = np.exp(x_shifted)
        a = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return a

    def backward(self, din):
        return din
