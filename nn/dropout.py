import numpy as np
from .module import Module


class Dropout(Module):
    def __str__(self): 
        return "Dropout({:5.2f})".format(self.p)

    def __init__(self, p=0.5):
        self.p = p

    def fprop(self, x, train=True):
        if self.training:
            mask = np.random.random(x.shape) > (1-self.p)
            scale = 1.0 / (1-self.p)
            self.a = x*mask*scale
            return self.a
        else:
            return x

    def bprop(self, delta_in):
        delta_out = delta_in*self.a
        return delta_out

    def update_params(self, lr):
        pass


class GaussianDropout():
    raise NotImplementedError()

