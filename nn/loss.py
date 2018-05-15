import numpy as np
from .module import Module


class MeanSquaredLoss(Module):
    def __str__(self): 
        return "MeanSquaredLoss()"
    
    def forward(self, x, t):
        num_batches = x.shape[0]
        cost = 0.5 * (x-t)**2 / num_batches
        return np.mean(np.sum(cost, axis=-1))
        
    def backward(self, y, t):
        num_batches = y.shape[0]
        delta_out = (1.0/num_batches) * (y-t)
        return delta_out
        

class CrossEntropyLoss(Module):
    def __str__(self): 
        return "CrossEntropyLoss()"
    
    def forward(self, x, t):
        tol = 1e-8
        return np.mean(np.sum(-t * np.log(x + tol), axis=-1))
        
    def backward(self, y, t):
        num_batches = y.shape[0]
        delta_out = (1.0/num_batches) * (y-t)
        return delta_out
