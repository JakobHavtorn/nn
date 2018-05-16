import IPython
import numpy as np

from .module import Module


class MeanSquaredLoss(Module):
    def __str__(self): 
        return "MeanSquaredLoss()"
    
    def forward(self, y, t):
        num_batches = y.shape[0]
        cost = 0.5 * (y-t)**2 / num_batches
        return np.mean(np.sum(cost, axis=-1))
        
    def backward(self, y, t):
        num_batches = y.shape[0]
        delta_out = (1.0/num_batches) * (y-t)
        return delta_out
        

class CrossEntropyLoss(Module):
    """Cross Entropy Loss function for D class classification

        Shape
        -----
        Input: 
            (N, C) where N is batch size and C is number of classes.
        Target: 
            (N, C) where each row is a one-hot encoded vector
        Output: float
            Scalar loss.
    """

    def __str__(self): 
        return "CrossEntropyLoss()"
    
    def forward(self, y, t, eps=1e-8):
        return - np.mean(np.log(y[t] + eps), axis=-1)
        
    def backward(self, y, t):
        delta_out = (1.0 / y.shape[0]) * (y - t)
        return delta_out
