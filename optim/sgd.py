import numpy as np
from .optimizer import Optimizer
import IPython


class SGD(Optimizer):
    def __init__(self, model, lr=0.001, momentum=0, nesterov=False, dampening=0, weight_decay=0):
        kwargs = {'lr': lr, 'weight_decay': weight_decay}
        super(SGD, self).__init__(model, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening

    def step(self):
        for p in self.model.parameters():
            if p.grad is None:
                continue
            dp = p.grad
            if self.weight_decay != 0:
                dp += self.weight_decay
            if self.momentum != 0:
                IPython.embed()
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer'] = np.zeros(p.shape)
                    self.state[p]['momentum_buffer'] *= self.momentum
                    self.state[p]['momentum_buffer'] += dp
                else:
                    self.state[p]['momentum_buffer'] *= self.momentum
                    self.state[p]['momentum_buffer'] += (1-self.dampening) * dp
                if self.nesterov:
                    dp += self.momentum * dp
                else:
                    dp = self.state[p]['momentum_buffer']
            p.data -= self.lr * dp
