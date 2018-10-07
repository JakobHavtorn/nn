import numpy as np
import IPython

from .module import Module
from .parameter import Parameter
from .activation import Sigmoid, Tanh, ReLU


class RNN(Module):
    """Vanilla recurrent neural network layer.

    The single time step forward transformation is
        h[:,t+1] = tanh(Whh * h[:,t] + Whx * X[:,t] + bh)
    with the following dimensions
        X:    (T, D)
        h:    (T, H)
        Whx:  (H, D)
        Whh:  (H, H)
        bh:   (H)
    where
        D:  input dimension
        T:  input sequence length
        H:  hidden dimension
    
    Parameters
    ----------
    input_size : [type]
        [description]
    hidden_size : [type]
        [description]
    bias : [type]
        [description]
    nonlinearity : [type]
        [description]
    Returns
    -------
    [type]
        [description]
    """


    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=Tanh()):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.Whh = Parameter(np.zeros((hidden_size, hidden_size)))
        self.Whx = Parameter(np.zeros((hidden_size, input_size)))
        if bias:
            self.bh = Parameter(np.zeros(hidden_size))
        else:
            self.bh = None
        self.reset_parameters()

    def reset_parameters(self):
        stdhh = np.sqrt(1./self.hidden_size)
        stdhx = np.sqrt(1./self.input_size)
        self.Whh.data = np.random.uniform(-stdhh, stdhh, size=(self.hidden_size, self.hidden_size))
        self.Whx.data = np.random.uniform(-stdhx, stdhx, size=(self.hidden_size, self.input_size))
        if self.bh is not None:
            self.bh.data = np.zeros(self.hidden_size)

    def forward_step(self, x, h):
        """Compute state k from the previous state (sk) and current input (xk),
        by use of the input weights (wx) and recursive weights (wRec).
        """
        return self.nonlinearity.forward(self.Whh.data @ h + self.Whx.data @ x + self.bh.data)

    def forward(self, X):
        """Unfold the network and compute all state activations given the input X,
        and input weights (wx) and recursive weights (wRec).
        Return the state activations in a matrix, the last column S[:,-1] contains the
        final activations.
        """
        # Initialise the matrix that holds all states for all input sequences.
        # The initial state s0 is set to 0.
        h = np.zeros((X.shape[0]+1, self.hidden_size))
        # Use the recurrence relation defined by forward_step to update the 
        # states trough time.
        for t in range(0, X.shape[0]):
            # h[t] = Whh * h[t-1] + Whx * X[t]
            h[t+1,:] = self.forward_step(X[t,:], h[t,:])
        # Cache
        self.X = X
        self.h = h
        return h
    
    def backward_step(self, dh, x_cache, h_cache):
        """Compute a single backwards time step.
        """
        # Activation
        dh = self.nonlinearity.backward(dh, h_cache)
        # Gradient of the linear layer parameters (accumulate)
        self.Whh.grad += np.outer(dh, h_cache)
        self.Whx.grad += np.outer(dh, x_cache)
        if self.bh is not None:
            self.bh.grad += dh
        # Gradient at the output of the previous layer
        dh_prev = dh @ self.Whh.data.T
        return dh_prev

    def backward(self, dout):
        """Backpropagate the gradient computed at the output (dout) through the network.
        Accumulate the parameter gradients for `Whx` and `Whh` by for each layer by addition.
        Return the parameter gradients as a tuple, and the gradients at the output of each layer.
        """
        # Initialise the array that stores the gradients of the cost with respect to the states.
        dh = np.zeros((self.X.shape[0]+1, self.hidden_size))
        dh[-1,:] = dout
        for t in range(self.X.shape[0], 0, -1):
            dh[t-1,:] = self.backward_step(dh[t,:], self.X[t-1,:], self.h[t-1,:])
        return dh


class GRUCell(Module):
    def __init__(self):
        pass


class LSTMCell(Module):
    def __init__(self):
        pass

