import numpy as np
from .module import Module
import IPython


class Sigmoid(Module):
    def __str__(self): 
        return "Sigmoid()"
    
    def forward(self, x):
        self.a = 1.0 / (1 + np.exp(-x))
        return self.a
        
    def backward(self, din):
        delta_out = self.a * (1 - self.a) * din
        return delta_out
        
    def update_params(self, lr):
        pass
    

class Tanh(Module):
    def __str__(self):
        return "Tanh()"

    def forward(self, x):
        self.a = np.tanh(x)
        return self.a

    def backward(self, din):
        return (1 - self.a ** 2) * din

    def update_params(self, lr):
        pass


class ReLU(Module):
    def __str__(self):
        return "ReLU()"

    def forward(self, x):
        self.a = np.maximum(0, x)
        return self.a
        
    def backward(self, din):
        return din * (self.a > 0).astype(self.a.dtype)
        
    def update_params(self, lr):
        pass
    

class Softplus(Module):
    def __str__(self):
        return "Softplus()"

    def forward(self, x):
        self.g = np.exp(x) + 1
        self.a = np.log(g)
        return self.a

    def backward(self, din):
        return din * 1 - self.g ** (-1)

    def update_params(self, lr):
        pass


class Softmax(Module):
    def __str__(self): 
        return "Softmax()"
    
    def forward(self, x):
        x_shifted = x - np.max(x)
        x_exp = np.exp(x_shifted)
        self.a = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return self.a
        
    def backward(self, din):
        return din
        
    def update_params(self, lr):
        pass
