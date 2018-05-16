import numpy as np


class Optimizer(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.lr = kwargs.pop('lr')
        self.l1_weight_decay = kwargs.pop('l1_weight_decay')
        self.l2_weight_decay = kwargs.pop('l2_weight_decay')
        self.state = {}
        for p in self.model.parameters():
            self.state[p] = {}

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = np.zeros(p.shape)

    def step(self):
        raise NotImplementedError()
