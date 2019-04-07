import IPython
import numpy as np

from .module import Module
from .parameter import Parameter


class BatchNorm1D(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.gamma = Parameter(np.zeros(num_features))
            self.beta = Parameter(np.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None
        if self.track_running_stats:
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
        self.reset_parameters()

    def __str__(self): 
        return "BatchNorm({:d}, momentum={:3.2f}, affine={}, track={})".format(self.num_features, self.momentum, self.affine, self.track_running_stats)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean = np.zeros(self.running_mean.shape)
            self.running_var = np.ones(self.running_var.shape)
            self.num_batches_tracked = 0

    def reset_parameters(self):
        if self.affine:
            self.gamma.data = np.random.uniform(size=self.gamma.shape)
            self.beta.data = np.zeros(self.beta.shape)
        self.reset_running_stats()

    def update_cache(self, x, x_norm, batch_mean, batch_var):
        # Cache
        self.cache = dict(x=x)
        if self.affine:
            self.cache.update(dict(x_norm=x_norm))
        if self.training:
            self.cache.update(dict(batch_mean=batch_mean, batch_var=batch_var))

    def forward(self, x):
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        if self.training:
            # Compute batch mean and variance and normalize x
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)

            # Update running mean and variance
            if self.momentum is None:
                # Cumulative moving average
                momentum = 1 - 1.0 / self.num_batches_tracked
                self.running_mean = momentum * self.running_mean + (1 - momentum) * batch_mean
                self.running_var = momentum * self.running_var + (1 - momentum) * batch_var
            else:
                # Exponential moving average
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            batch_mean, batch_var = None, None
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        if self.affine:
            x_out = x_norm * self.gamma.data + self.beta.data
        else:
            x_out = x_norm
        # Cache
        self.update_cache(x, x_norm, batch_mean, batch_var)
        return x_out

    def backward(self, dout):
        x, batch_mean, batch_var = self.cache['x'], self.cache['batch_mean'], self.cache['batch_var']
        N, _ = dout.shape
        dx_norm = dout * self.gamma.data
        dsample_var = np.sum(dx_norm * (x - batch_mean) * (-0.5) * (batch_var + self.eps)**(-1.5), axis=0)

        dsample_mean = np.sum(dx_norm * (-1/np.sqrt(batch_var + self.eps)) , axis=0) + dsample_var * ((np.sum(-2*(x - batch_mean))) / N)

        dx = dx_norm * (1/np.sqrt(batch_var + self.eps)) + dsample_var * (2*(x - batch_mean)/N) + dsample_mean/N
        if self.affine:
            x_norm = self.cache['x_norm']
            self.beta.grad = np.sum(dout, axis=0)
            self.gamma.grad = np.sum(dout * x_norm, axis=0)
        return dx
