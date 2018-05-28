from .module import Module
import IPython


class Flatten(Module):
    """Flattens the batch.

    During the forward pass, flattens an input with some dimensionality to a batch matrix,
        (N, d1, d2, ..., dM) --> (N, D)
    where
        D = d1 * d2 * ... * dM

    During the backwards pass, the inverse operation is applied
        (N, D) --> (N, d1, d2, ..., dM)
    where (N, d1, d2, ..., dM) is stored in cache.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def __str__(self): 
        return "Flatten()"

    def forward(self, x):
        self.forward_shape = x.shape[1:]
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(dout.shape[0], *self.forward_shape)
