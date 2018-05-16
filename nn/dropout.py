import numpy as np

from .module import Module


class Dropout(Module):
    """Dropout layer which randomly sets activations to zero.
    
    Parameters
    ----------
    p : float
        The probability of zeroing any activation. Must be smaller than 1 and larger than 0
    """
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def __str__(self): 
        return "Dropout({:.2f})".format(self.p)

    def forward(self, x, train=True):
        if self.training:
            mask = np.random.random(x.shape) > self.p
            scale = 1.0 / (1-self.p)
            self.a = x*mask*scale
            return self.a
        else:
            return x

    def backward(self, delta_in):
        delta_out = delta_in*self.a
        return delta_out

    def update_params(self, lr):
        pass
