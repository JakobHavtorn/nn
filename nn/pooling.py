import IPython
import numpy as np

from .im2col import col2im_indices, im2col_indices
from .module import Module


class _Pooling(Module):
    def __init__(self, kernel_size, stride, padding):
        super(_Pooling, self).__init__()
        # assert type(kernel_size) is tuple, "Please specifiy kernel size in each dimension as a tuple."
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, X):
        # Shape
        N, C, H, W = X.shape
        h_out = (H - self.kernel_size) / self.stride + 1
        w_out = (W - self.kernel_size) / self.stride + 1
        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')
        h_out, w_out = int(h_out), int(w_out)
        # Reshape
        X_reshaped = X.reshape(N * C, 1, H, W)
        X_col = im2col_indices(X_reshaped, self.kernel_size, self.kernel_size, padding=0, stride=self.stride)
        # Pool
        out = self._pool(X_col)
        # Reshape
        out = out.reshape(h_out, w_out, N, C)
        out = out.transpose(2, 3, 0, 1)
        return out

    def backward(self, dout, cache):
        # Shapes
        IPython.embed()
        N, C, W, H = X.shape
        dX_col = np.zeros_like(X_col)
        dout_col = dout.transpose(2, 3, 0, 1).ravel()
        # Backwards pool
        dX = self._dpool(dX_col, dout_col)
        # Reshape
        dX = col2im_indices(dX_col, (N * C, 1, H, W), self.kernel_size, self.kernel_size, padding=0, stride=self.stride)
        dX = dX.reshape(X.shape)
        return dX


class MaxPool2D(_Pooling):
    """Pooling module which performs the two-dimensional max pooling operation.

    The dimensions are as for the two-dimensional convolution.
    
    Parameters
    ----------
    in_channels : tuple
        The number of input channels
    out_channels : tuple
        The number of kernels, or feature maps, to learn
    kernel_size : tuple
        The dimensions of the kernel
    stride : int
        The convolution stride
    padding : int
        The zero padding to be applied to the input
    bias : bool
        Whether or not to use bias
    """

    def __init__(self, kernel_size, stride=2, padding=0):
        super(MaxPool2D, self).__init__(kernel_size, stride=stride, padding=padding)

    def _pool(self, X_col):
        self.max_idx = np.argmax(X_col, axis=0)
        out = X_col[self.max_idx, range(self.max_idx.size)]
        return out

    def _dpool(self, dX_col, dout_col):
        dX_col[self.max_idx, range(dout_col.size)] = dout_col
        return dX_col


class AvgPool2D(_Pooling):
    """Pooling module which performs the two-dimensional average pooling operation.

    The dimensions are as for the two-dimensional convolution.
    
    Parameters
    ----------
    in_channels : tuple
        The number of input channels
    out_channels : tuple
        The number of kernels, or feature maps, to learn
    kernel_size : tuple
        The dimensions of the kernel
    stride : int
        The convolution stride
    padding : int
        The zero padding to be applied to the input
    bias : bool
        Whether or not to use bias
    """

    def __init__(self, kernel_size, stride=2, padding=0):
        super(AvgPool2D, self).__init__()

    def _pool(X_col):
        out = np.mean(X_col, axis=0)
        return out

    def _dpool(dX_col, dout_col):
        dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
        return dX_col


def avgpool_forward(X, size=2, stride=2):
    def avgpool(X_col):
        out = np.mean(X_col, axis=0)
        cache = None
        return out, cache

    return _pool_forward(X, avgpool, size, stride)


def avgpool_backward(dout, cache):
    def davgpool(dX_col, dout_col, pool_cache):
        dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
        return dX_col

    return _pool_backward(dout, davgpool, cache)
