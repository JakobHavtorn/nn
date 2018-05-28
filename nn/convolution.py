import IPython
import numpy as np

from .im2col import col2im_indices, im2col_indices
from .module import Module
from .parameter import Parameter


class Conv2D(Module):
    """Convolution module which performs the two-dimensional convolution.

    The forward transformation is
        S = X * K
    with the following dimensions
        X:  (N, C, H, W)
        K:  (NK, C, HK, WK)
        b:  (NK)
    where
        N:  batch size
        C:  number of image channels
        H:  height of image
        W:  width of the image
        NK: number of kernels in the feature map K
        HK: height of the kernel
        WK: width of the kernel

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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2D, self).__init__()
        assert type(kernel_size) is tuple, "Please specifiy kernel size in each dimension as a tuple."
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.padding = int(padding)
        self.kernel_size = kernel_size
        self.K = Parameter(np.zeros((out_channels, in_channels, *self.kernel_size)))
        if bias:
            self.b = Parameter(np.zeros(((self.out_channels, 1))))
        else:
            self.b = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / np.sqrt(n)
        self.K.data = np.random.uniform(-stdv, stdv, self.K.shape)
        if self.b is not None:
            self.b.data = np.random.uniform(-stdv, stdv, self.b.shape)

    def forward(self, X):
        # Compute output dimensions
        N, C, H, W = X.shape
        h_out = (H - self.kernel_size[0] + 2 * self.padding) / self.stride + 1
        w_out = (W - self.kernel_size[1] + 2 * self.padding) / self.stride + 1
        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')
        h_out, w_out = int(h_out), int(w_out)
        # Reshape
        X_col = im2col_indices(X, self.kernel_size[0], self.kernel_size[1], padding=self.padding, stride=self.stride)
        K_col = self.K.data.reshape(self.out_channels, -1)
        # Convolve
        S = K_col @ X_col
        if self.b is not None:
            S += self.b.data
        # Reshape
        S = S.reshape(self.out_channels, h_out, w_out, N)
        S = S.transpose(3, 0, 1, 2)
        # Cache
        self.X = X
        self.X_col = X_col
        return S

    def backward(dout):
        IPython.embed()
        # Bias gradient
        self.b.grad = np.sum(dout, axis=(0, 2, 3))
        self.b.grad = self.b.grad.reshape(self.out_channels, -1)
        # Kernel gradient
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        self.K.grad = dout_reshaped @ self.X_col.T
        self.K.grad = self.K.grad.reshape(self.K.shape)
        # Input gradient
        W_reshape = W.reshape(self.out_channels, -1)
        dself.X_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col, X.shape, self.kernel_size[0], self.kernel_size[1], padding=self.padding, stride=self.stride)
        return dX
