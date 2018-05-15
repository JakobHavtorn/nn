import numpy as np


class Parameter(object):
    def __init__(self, data):
        self.data = data
        self.grad = None
    
    def __str__(self):
        s = "Parameter containing:\n"
        s += self.data.__repr__()
        if self.grad is not None:
            s += "\nand gradient\n"
            s += self.grad.__repr__()
        return s

    def shape(self):
        return self.data.shape
