import numpy as np


class Sigmoid():
    def __str__(self): 
        return "Sigmoid()"
    
    def forward(self, x):
        self.a = 1.0 / (1 + np.exp(-x))
        return self.a
        
    def backward(self, delta_in):
        delta_out = self.a * (1 - self.a) * delta_in
        return delta_out
        
    def update_params(self, lr):
        pass
    

class Tanh():
    def __str__(self):
        return "Tanh()"

    def forward(self, x):
        self.a = np.tanh(x)
        return self.a

    def backward(self, delta_in):
        return (1 - self.a ** 2) * delta_in

    def update_params(self, lr):
        pass


class ReLU():
    def __str__(self):
        return "ReLU()"

    def forward(self, x):
        self.a = np.maximum(0, x)
        return self.a
        
    def backward(self, delta_in):
        return delta_in * (self.a > 0).astype(self.a.dtype)
        
    def update_params(self, lr):
        pass
    

class Softplus():
    def __str__(self):
        return "Softplus()"

    def forward(self, x):
        self.g = np.exp(x) + 1
        self.a = np.log(g)
        return self.a

    def backward(self, delta_in):
        return delta_in * 1 - self.g ** (-1)

    def update_params(self, lr):
        pass


class Softmax():
    def __str__(self): 
        return "Softmax()"
    
    def forward(self, x):
        x_exp = np.exp(x)
        normalizer = x_exp.sum(axis=-1, keepdims=True)
        self.a = x_exp / normalizer
        return self.a
        
    def backward(self, delta_in):
        return delta_in
        
    def update_params(self, lr):
        pass
