import numpy as np
from .module import Module
from .parameter import Parameter


class BatchNorm1D(Module):
    "https://wiseodd.github.io/techblog/2016/07/04/batchnorm/"
    
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

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean = np.zeros(self.running_mean.shape)
            self.running_var = np.ones(self.running_var.shape)
            self.num_batches_tracked = 0

    def reset_parameters(self):
        if self.affine:
            self.gamma.data = np.random.uniform(self.gamma.shape)
            self.beta.data = np.zeros(self.beta.shape)
        self.reset_running_stats()

    def forward(self, x):
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        if self.training:
            # Compute batch mean and variance and normalize x
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)
            self.x_norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            if self.affine:
                x_out = self.x_norm * self.gamma + self.beta
            else:
                x_out = self.x_norm
            # Update running mean and variance
            if self.momentum is None:
                # Cumulative moving average
                momentum = 1 - 1.0 / self.num_batches_tracked
                self.running_mean = momentum * self.running_mean + (1 - momentum) * self.batch_mean
                self.running_var = momentum * self.running_var + (1 - momentum) * self.batch_var
            else:
                # Exponential moving average
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            # Cache this batch
            self.x = x
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var)
            if self.affine:
                x_out = x_norm * self.gamma + self.beta
            else:
                x_out = x_norm
        return x_out

    def backward(self, dout):
        N, _ = dout.shape
        dx_norm = dout * self.gamma
        dsample_var = np.sum(dx_norm * (self.x - self.batch_mean) * (-0.5) * (self.batch_var + self.eps)**(-1.5), axis=0)
        dsample_mean = np.sum(dx_norm * (-1/np.sqrt(self.batch_var + self.eps)) , axis=0) + dsample_var * ((np.sum(-2*(self.x - self.batch_var))) / N)
        dx = dx_norm * (1/np.sqrt(self.batch_mean + self.eps)) + dsample_var * (2*(self.x-self.batch_var)/N) + dsample_mean/N
        if self.affine:
            self.beta.grad = np.sum(dout, axis=0)
            self.gamma.grad = np.sum(dout * self.x_norm, axis=0)
        return dx
