import IPython
import numpy as np

from .im2col import col2im_indices, im2col_indices
from .module import Module

try:
    from .im2col_cython import col2im_cython, im2col_cython
    from .im2col_cython import col2im_6d_cython
except ImportError:
    print("Failed to import im2col and col2im Cython versions.")
    print('Run the following from the nn directory and try again:')
    print('python setup.py build_ext --inplace')


class _Pooling(Module):
    def __init__(self, kernel_size, stride, padding):
        super(_Pooling, self).__init__()
        # assert type(kernel_size) is tuple, "Please specifiy kernel size in each dimension as a tuple."
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size[0]
        self.padding = padding
        if stride is not None:
            self.stride = stride
        elif kernel_size[0] == kernel_size[1]:
            self.stride = kernel_size[0]
        else:
            raise ValueError("Stride can only default to kernel size if kernel is square. Had `kernel_size={}".format(kernel_size))

    def forward(self, X):
        # Shape
        N, C, H, W = X.shape
        h_out = (H - self.kernel_size[0]) / self.stride + 1
        w_out = (W - self.kernel_size[1]) / self.stride + 1
        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')
        h_out, w_out = int(h_out), int(w_out)
        # Reshape
        X_reshaped = X.reshape(N * C, 1, H, W)
        X_col = im2col_cython(X_reshaped, self.kernel_size[0], self.kernel_size[1], padding=self.padding, stride=self.stride)
        # X_col = im2col_indices(X_reshaped, self.kernel_size[0], self.kernel_size[1], padding=self.padding, stride=self.stride)
        # Pool
        out = self._pool(X_col)
        # Reshape
        out = out.reshape(h_out, w_out, N, C)
        out = out.transpose(2, 3, 0, 1)
        # Cache
        self.X_shape = X.shape
        self.X_col = X_col
        return out

    def backward(self, dout):
        # Shapes
        N, C, W, H = self.X_shape
        # Backwards pool
        dout_col = dout.transpose(2, 3, 0, 1).ravel()
        dX_col = np.zeros_like(self.X_col)
        dX_col = self._dpool(dX_col, dout_col)
        # Reshape
        dX = col2im_cython(dX_col, N * C, 1, H, W, self.kernel_size[0], self.kernel_size[1], padding=self.padding, stride=self.stride)
        # dX = col2im_indices(dX_col, (N * C, 1, H, W), self.kernel_size[0], self.kernel_size[1], padding=self.padding, stride=self.stride)
        dX = dX.reshape(self.X_shape)
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
        The pooling stride
    padding : int
        The zero padding to be applied to the input
    bias : bool
        Whether or not to use bias
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2D, self).__init__(kernel_size, stride=stride, padding=padding)

    def __str__(self): 
        return "MaxPool2D(kernel=({:d},{:d}), stride={:d}, padding={:d})".format(self.kernel_size[0], self.kernel_size[1], self.stride, self.padding)

    def _pool(self, X_col):
        self.max_idx = np.argmax(X_col, axis=0)
        out = X_col[self.max_idx, range(self.max_idx.size)]
        return out

    def _dpool(self, dX_col, dout_col):
        # dX_col[self.max_idx, range(dout_col.size)] = dout_col
        dX_col[self.max_idx, np.arange(dX_col.shape[1])] = dout_col
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
        The pooling stride
    padding : int
        The zero padding to be applied to the input
    bias : bool
        Whether or not to use bias
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgPool2D, self).__init__()

    def __str__(self): 
        return "AvgPool2D(kernel=({:d},{:d}), stride={:d}, padding={:d})".format(self.kernel_size[0], self.kernel_size[1], self.stride, self.padding)

    def _pool(self, X_col):
        out = np.mean(X_col, axis=0)
        return out

    def _dpool(self, dX_col, dout_col):
        dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
        return dX_col
