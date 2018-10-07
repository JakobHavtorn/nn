import IPython
import numpy as np

from .module import Module


class MeanSquaredLoss(Module):
    """Mean Squared Loss function for multiple regression.

    Shapes
    ------
    Input: 
        (N, C) where N is batch size and C is number of classes.
    Target: 
        (N, C) where each row is a one-hot encoded vector
    Output: float
        Scalar loss.
    """
    def __str__(self): 
        return "MeanSquaredLoss()"
    
    def forward(self, y, t):
        N = y.shape[0]
        errors =  np.linalg.norm(y-t, 2, axis=0)
        MSE = 0.5 * np.sum(errors**2) / N
        return MSE
        
    def backward(self, y, t):
        N = y.shape[0]
        delta_out = np.sum(y-t, axis=0) / N
        return delta_out
        

class CrossEntropyLoss(Module):
    """Cross Entropy Loss function for D class classification.

        Shapes
        ------
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
