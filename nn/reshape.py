from .module import Module
import IPython


class Flatten(Module):
    """Flattens the batch.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(dout.shape[0], -1)
