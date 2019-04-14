import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), l1_weight_decay=0.0, l2_weight_decay=0.0,
                 gradient_clip=0.0, amsgrad=False, eps=1e-8):
        kwargs = {'lr': lr, 'l1_weight_decay': l1_weight_decay, 'l2_weight_decay': l2_weight_decay,
                  'gradient_clip': gradient_clip}
        super().__init__(parameters, **kwargs)
        self.betas = betas
        self.amsgrad = amsgrad
        self.eps = eps

    def _step(self):
        beta1, beta2 = self.betas
        for p in self.parameters():
            # Get gradient and state
            if p.grad is None:
                continue
            dp = p.grad

            # State initialization
            if len(self.state[p]) == 0:
                self.state[p]['step'] = 0  # Step
                self.state[p]['exp_avg'] = np.zeros_like(p.data)  # Exponential moving average of gradient
                self.state[p]['exp_avg_sq'] = np.zeros_like(p.data)  # Exponential moving average of squared gradient
                self.state[p]['max_exp_avg_sq'] = np.zeros_like(p.data) if self.amsgrad else None  # Max of all ma

            # Retrieve self.state[p] variables
            exp_avg, exp_avg_sq = self.state[p]['exp_avg'], self.state[p]['exp_avg_sq']
            max_exp_avg_sq = self.state[p]['max_exp_avg_sq'] if self.amsgrad else None
            self.state[p]['step'] += 1

            # Weight decay
            if self.l1_weight_decay != 0:
                dp += self.l1_weight_decay
            if self.l2_weight_decay != 0:
                dp += self.l2_weight_decay * p.data

            # Decay the first and second moment running average coefficient
            exp_avg = beta1 * exp_avg + (1 - beta1) * dp
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (dp ** 2)
            if self.amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                np.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max for normalizing running avg of gradient
                max_exp_avg_sq = np.sqrt(max_exp_avg_sq + self.eps)
                denom = max_exp_avg_sq
            else:
                exp_avg_sq = np.sqrt(exp_avg_sq + self.eps)
                denom = exp_avg_sq

            # Compute step size
            bias_correction1 = 1 - beta1 ** self.state[p]['step']
            bias_correction2 = 1 - beta2 ** self.state[p]['step']
            step_size = self.lr * np.sqrt(bias_correction2) / bias_correction1

            # Update parameters
            p.data -= step_size * exp_avg / denom
