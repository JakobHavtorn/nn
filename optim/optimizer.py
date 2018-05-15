import numpy as np


class Optimizer(object):
    def __init__(self, module):
        self.module = module
        for p in self.parameters():
            self.state[p] = {}

    def parameters(self):


    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad = np.zeros(p.shape)

    def step(self):
        raise NotImplementedError()

    
