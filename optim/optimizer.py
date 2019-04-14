import numpy as np


class Optimizer(object):
    """Base class for all optimizers
    """

    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        self.lr = kwargs.pop('lr', 0.001)
        self.l1_weight_decay = kwargs.pop('l1_weight_decay', 0.0)
        self.l2_weight_decay = kwargs.pop('l2_weight_decay', 0.0)
        self.gradient_clip = kwargs.pop('gradient_clip', 0.0)
        self.state = dict()
        for p in self.parameters():
            self.state[p] = dict()

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad = np.zeros(p.shape)

    def clip_grads(self):
        if self.gradient_clip != 0.0:
            for p in self.parameters():
                p.grad = np.clip(p.grad, -self.gradient_clip, self.gradient_clip)

    def step(self):
        self.clip_grads()
        self._step()

    def _step(self):
        raise NotImplementedError()
        