import IPython
import numpy as np

from .module import Module


# TODO Add reduction methods somehow


class Loss(Module):
    def __init__(self, reduction=None):
        self.reduction = reduction


class MeanSquaredLoss(Loss):
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
    def __init__(self, reduction=None):
        super().__init__(reduction)

    def __str__(self): 
        return f'MeanSquaredLoss(reduction={self.reduction})'

    def forward(self, y, t):
        loss = 0.5 * ((y - t) ** 2)
        if self.reduction is not None:
            loss = self.reduction(loss)
        return loss

    def backward(self, y, t):
        N = y.shape[0]
        delta_out = np.sum(y-t, axis=0) / N
        return delta_out


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss function for C class classification.

        Shapes
        ------
        Input: 
            (N, C) where N is batch size and C is number of classes.
        Target: 
            (N, C) where each row is a one-hot encoded vector
        Output: float
            Scalar loss.
    """
    def __init__(self, reduction=None, eps=1e-8):
        super().__init__(reduction)
        self.eps = eps

    def __str__(self):
        return f'CrossEntropyLoss(reduction={self.reduction}, eps={self.eps})'

    def forward(self, y, t):
        loss = -np.log(y[t.astype(bool)] + self.eps)
        if self.reduction is not None:
            loss = self.reduction(loss)
        return loss

    def backward(self, y, t):
        delta_out = (1.0 / y.shape[0]) * (y - t)
        return delta_out
