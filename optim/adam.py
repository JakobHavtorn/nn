import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, model, lr=0.001, momentum=0.0, nesterov=False, dampening=0.0, l1_weight_decay=0.0, l2_weight_decay=0.0, gradient_clip=0.0):
        kwargs = {'lr': lr, 'l1_weight_decay': l1_weight_decay, 'l2_weight_decay': l2_weight_decay, 'gradient_clip': gradient_clip}
        super().__init__(model, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening

    def _step(self):
        for p in self.model.parameters():
            if p.grad is None:
                continue
            dp = p.grad
            # if self.l1_weight_decay != 0:
            #     dp += self.l1_weight_decay
            # if self.l2_weight_decay != 0:
            #     dp += self.l2_weight_decay * p.data
            # if self.momentum != 0:
            #     if 'momentum_buffer' not in self.state[p]:
            #         self.state[p]['momentum_buffer'] = np.zeros(p.shape)
            #         self.state[p]['momentum_buffer'] += dp
            #     else:
            #         self.state[p]['momentum_buffer'] *= self.momentum
            #         self.state[p]['momentum_buffer'] += (1-self.dampening) * dp
            #     if self.nesterov:
            #         dp += self.momentum * dp
            #     else:
            #         dp = self.state[p]['momentum_buffer']
            p.data -= self.lr * dp
