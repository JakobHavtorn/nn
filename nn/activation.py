import numpy as np

from .module import Module


class Sigmoid(Module):
    def __str__(self): 
        return "Sigmoid()"
    
    def forward(self, x):
        self.a = 1.0 / (1 + np.exp(-x))
        return self.a
        
    def backward(self, din, a_cache=None):
        a_cache = self.a if a_cache is None else a_cache
        return a_cache * (1 - a_cache) * din


class Tanh(Module):
    def __str__(self):
        return "Tanh()"

    def forward(self, x):
        self.a = np.tanh(x)
        return self.a

    def backward(self, din, a_cache=None):
        a_cache = self.a if a_cache is None else a_cache
        return (1 - a_cache ** 2) * din


class ReLU(Module):
    def __str__(self):
        return "ReLU()"

    def forward(self, x):
        self.a = np.maximum(0, x)
        return self.a
        
    def backward(self, din, a_cache=None):
        a_cache = self.a if a_cache is None else a_cache
        return din * (a_cache > 0).astype(a_cache.dtype)
        

class Softplus(Module):
    def __str__(self):
        return "Softplus()"

    def forward(self, x):
        self.g = np.exp(x) + 1
        return np.log(self.g)

    def backward(self, din, g_cache=None):
        g_cache = self.g if g_cache is None else g_cache
        return din * 1 - g_cache ** (-1)


class Softmax(Module):
    def __str__(self): 
        return "Softmax()"
    
    def forward(self, x):
        x_shifted = x - np.max(x)
        x_exp = np.exp(x_shifted)
        self.a = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return self.a
        
    def backward(self, din, cache=None):
        return din
        