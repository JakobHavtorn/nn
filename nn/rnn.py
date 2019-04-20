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
        X:    (T, N, D)
        h:    (N, H)
        Whx:  (H, D)
        Whh:  (H, H)
        b:    (H)
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


    def __init__(self, input_size, hidden_size, output_size, bias=True, nonlinearity=Tanh(), time_first=True, bptt_truncate=0):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.bptt_truncate = bptt_truncate
        self.Wxh = Parameter(np.zeros((hidden_size, input_size)))
        self.Whh = Parameter(np.zeros((hidden_size, hidden_size)))
        self.Why = Parameter(np.zeros((output_size, hidden_size)))
        if bias:
            self.bh = Parameter(np.zeros(hidden_size))
            self.by = Parameter(np.zeros(output_size))
        else:
            self.bh = None
            self.by = None
        self.time_first = time_first
        if time_first:
            self.t_dim = 0
            self.n_dim = 1
            self.d_dim = 2
        else:
            self.t_dim = 1
            self.n_dim = 0
            self.d_dim = 2
        self.reset_parameters()

    def reset_parameters(self):
        stdhh = np.sqrt(1. / self.hidden_size)
        stdhx = np.sqrt(1. / self.input_size)
        stdhy = np.sqrt(1. / self.output_size)
        self.Wxh.data = np.random.uniform(-stdhx, stdhx, size=(self.hidden_size, self.input_size))
        self.Whh.data = np.random.uniform(-stdhh, stdhh, size=(self.hidden_size, self.hidden_size))
        self.Why.data = np.random.uniform(-stdhy, stdhy, size=(self.output_size, self.hidden_size))
        if self.bh is not None:
            self.bh.data = np.zeros(self.hidden_size)
            self.by.data = np.zeros(self.output_size)

    def forward_step(self, x, h):
        """Compute state k from the previous state (sk) and current input (xk),
        by use of the input weights (wx) and recursive weights (wRec).
        """
        # import IPython
        # IPython.embed()
        return self.nonlinearity.forward(h @ self.Whh.data.T + x @ self.Wxh.data.T + self.b.data)

    def forward(self, X, h0=None):
        """Unfold the network and compute all state activations given the input X,
        and input weights (wx) and recursive weights (wRec).
        Return the state activations in a matrix, the last column S[:,-1] contains the
        final activations.
        """
        # Initialise the matrix that holds all states for all input sequences.
        # The initial state s0 is set to 0.
        if not self.time_first:
            X = X.transpose(self.n_dim, self.t_dim, self.n_dim)  # [N, T, D] --> [T, N, D]
        h = np.zeros((X.shape[self.t_dim] + 1, X.shape[self.n_dim], self.hidden_size))  # (T, N, H)
        if h0:
            h[0] = h0
        # Use the recurrence relation defined by forward_step to update the states trough time.
        for t in range(0, X.shape[self.t_dim]):

            # W x 
            h[t + 1] = self.nonlinearity.forward(np.dot(X[t], self.Wxh.data.T) + np.dot(h[t], self.Whh.data.T) + self.b.data)
            
            # h[t + 1] = self.forward_step(X[t, :], h[t])

            # np.dot(self.Wxh.data, X[t][5])
            # np.dot(X[t], self.Wxh.data.T)

        # Cache
        self.X = X
        self.h = h
        return h

    def backward_step_old_broken(self, dh, x_cache, h_cache):
        """Compute a single backwards time step.
        """
        # https://gist.github.com/karpathy/d4dee566867f8291f086

        # Activation
        dh = self.nonlinearity.backward(dh, h_cache)
        # Gradient of the linear layer parameters (accumulate)
        self.Whh.grad += dh.T @ h_cache  # np.outer(dh, h_cache)
        self.Wxh.grad += dh.T @ x_cache  # np.outer(dh, x_cache)
        if self.b is not None:
            self.b.grad += dh.sum(axis=0)
        # Gradient at the output of the previous layer
        dh_prev = dh @ self.Whh.data.T  # self.Whh.data @ dh.T
        return dh_prev

    def backward_old_broken(self, delta):
        """Backpropagate the gradient computed at the output (delta) through the network.
        Accumulate the parameter gradients for `Whx` and `Whh` by for each layer by addition.
        Return the parameter gradients as a tuple, and the gradients at the output of each layer.
        """
        # Initialise the array that stores the gradients of the cost with respect to the states.
        dh = np.zeros((self.X.shape[self.t_dim] + 1, self.X.shape[self.n_dim], self.hidden_size))
        dh[-1] = delta
        for t in range(self.X.shape[self.t_dim], 0, -1):
            dh[t - 1, :] = self.backward_step_old_broken(dh[t, :], self.X[t - 1, :], self.h[t - 1, :])
        return dh

    def backward(self, delta):
        """Backpropagate the gradient computed at the output (delta) through the network.
        Accumulate the parameter gradients for `Whx` and `Whh` by for each layer by addition.
        Return the parameter gradients as a tuple, and the gradients at the output of each layer.
        """
        # http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
        # Initialise the array that stores the gradients of the cost with respect to the states.
        # dh = np.zeros((self.X.shape[self.t_dim] + 1, self.X.shape[self.n_dim], self.hidden_size))
        # dh[-1] = delta
        dh_t = delta
        dLdz = delta
        for t in range(self.X.shape[self.t_dim], 0, -1):

            # IPython.embed()
            # Initial delta calculation: dL/dz (TODO Don't really care about this)
            # dLdz = self.V.T.dot(delta_o[t]) * (1 - (self.h[t] ** 2))   # (1 - (self.h[t] ** 2)) is Tanh()
            dh_t = self.nonlinearity.backward(dh_t, self.h[t])
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:  # TODO Can we maybe vectorize this loop?
                # print &quot;Backpropagation step t=%d bptt step=%d &quot; % (t, bptt_step)
                # Add to gradients at each previous step
                self.Whh.grad += np.einsum('NH,iNH->NH', dh_t, self.h[bptt_step - 1])
                # self.Whh.grad += np.outer(dh_t, self.h[bptt_step - 1])
                self.Wxh.grad[:, self.X[bptt_step]] += dh_t
                # self.Wxh.grad[:, self.X[bptt_step]] += dLdz  # TODO Really want dh/dU
                # Update delta for next step dL/dz at t-1
                dh_t = self.nonlinearity.backward(self.Whh.data.T.dot(dh_t), self.h[bptt_step-1])  # (1 - self.h[bptt_step-1] ** 2)


            # dh[t - 1, :] = self.backward_step(dh[t, :], self.X[t - 1, :], self.h[t - 1, :])
        return dh_t

    def backward_step(self, dh, x_cache, h_cache):
        pass
        # return [dLdU, dLdV, dLdW]

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.Wxh.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.Whh.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation: dL/dz
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))   # (1 - (s[t] ** 2)) is Tanh()
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print &quot;Backpropagation step t=%d bptt step=%d &quot; % (t, bptt_step)
                # Add to gradients at each previous step
                dLdW += np.outer(delta_t, s[bptt_step - 1])              
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step dL/dz at t-1
                delta_t = self.Whh.data.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]


# http://willwolf.io/2016/10/18/recurrent-neural-network-gradients-and-lessons-learned-therein/
# https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf
# https://peterroelants.github.io/posts/rnn-implementation-part01/
# http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/


class GRU(Module):
    def __init__(self):
        pass


class LSTM(Module):
    def __init__(self, input_size, hidden_size=128, bias=True, time_first=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_first = time_first
        if time_first:
            self.t_dim = 0
            self.n_dim = 1
            self.d_dim = 2
        else:
            self.t_dim = 1
            self.n_dim = 0
            self.d_dim = 2

        D = self.input_size
        H = self.hidden_size
        Z = D + H  # Concatenation
        self.Wf = Parameter(np.zeros((Z, H)))
        self.Wi = Parameter(np.zeros((Z, H)))
        self.Wc = Parameter(np.zeros((Z, H)))
        self.Wo = Parameter(np.zeros((Z, H)))
        self.Wy = Parameter(np.zeros((H, D)))
        if bias:
            self.bf = Parameter(np.zeros((1, H)))
            self.bi = Parameter(np.zeros((1, H)))
            self.bc = Parameter(np.zeros((1, H)))
            self.bo = Parameter(np.zeros((1, H)))
            self.by = Parameter(np.zeros((1, D)))
        else:
            self.bf = None
            self.bi = None
            self.bc = None
            self.bo = None
            self.by = None
        
        self.reset_parameters()

    def reset_parameters(self):
        D = self.input_size
        H = self.hidden_size
        Z = D + H  # Concatenation
        self.Wf.data = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.Wi.data = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.Wc.data = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.Wo.data = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.Wy.data = np.random.randn(H, D) / np.sqrt(D / 2.)
        if self.bf is not None:
            self.bf.data = np.zeros((1, H))
            self.bi.data = np.zeros((1, H))
            self.bc.data = np.zeros((1, H))
            self.bo.data = np.zeros((1, H))
            self.by.data = np.zeros((1, D))
        else:
            self.bf = None
            self.bi = None
            self.bc = None
            self.bo = None
            self.by = None
        self.sigmoidf = Sigmoid()
        self.sigmoidi = Sigmoid()
        self.sigmoido = Sigmoid()
        self.tanhc = Tanh()
        self.tanh = Tanh()

    def forward_step(self, x, state):
        h_old, c_old = state

        # # One-hot encode
        # X_one_hot = np.zeros(D)
        # X_one_hot[X] = 1.
        # X_one_hot = X_one_hot.reshape(1, -1)

        # Concatenate old state with current input
        hx = np.column_stack((h_old, x))

        hf = self.sigmoidf.forward(hx @ self.Wf.data + self.bf.data)
        hi = self.sigmoidi.forward(hx @ self.Wi.data + self.bi.data)
        ho = self.sigmoido.forward(hx @ self.Wo.data + self.bo.data)
        hc = self.tanhc.forward(hx @ self.Wc.data + self.bc.data)

        c = hf * c_old + hi * hc
        h = ho * self.tanh.forward(c)

        # y = h @ Wy + by
        # prob = softmax(y)

        self.cache = dict(hx=[*self.cache['hx'], hx],
                          hf=[*self.cache['hf'], hf],
                          hi=[*self.cache['hi'], hi],
                          ho=[*self.cache['ho'], ho],
                          hc=[*self.cache['hc'], hc],
                          c=[*self.cache['c'], c],
                          c_old=[*self.cache['c_old'], c_old])
        return (h, c)

    def forward(self, X):
        self.cache = dict(hx=[],
                          hf=[],
                          hi=[],
                          ho=[],
                          hc=[],
                          c=[],
                          c_old=[])
        if not self.time_first:
            X = X.transpose(self.n_dim, self.t_dim, self.n_dim)  # [N, T, D] --> [T, N, D]
        h = np.zeros((X.shape[self.t_dim] + 1, X.shape[self.n_dim], self.hidden_size))  # (T, N, H)
        c = np.zeros((X.shape[self.t_dim] + 1, X.shape[self.n_dim], self.hidden_size))  # (T, N, H)
        # Use the recurrence relation defined by forward_step to update the states trough time.
        for t in range(0, X.shape[self.t_dim]):
            h[t + 1], c[t + 1] = self.forward_step(X[t, :], (h[t], c[t]))
        return h[-1]

    def backward_step(self, dh_next, dc_next, t):
        # Unpack the cache variable to get the intermediate variables used in forward step
        hx = self.cache['hx'][t]
        hf = self.cache['hf'][t]
        hi = self.cache['hi'][t]
        ho = self.cache['ho'][t]
        hc = self.cache['hc'][t]
        c = self.cache['c'][t]
        c_old = self.cache['c_old'][t]
        IPython.embed()

        # # Softmax loss gradient
        # dy = prob.copy()
        # dy[1, y_train] -= 1.

        # # Hidden to output gradient
        # dWy = h.T @ dy
        # dby = dy
        # # Note we're adding dh_next here
        # dh = dy @ Wy.T + dh_next

        # Gradient for ho in h = ho * tanh(c)
        dho = self.tanh.forward(c) * dh_next
        dho = self.sigmoido.backward(ho) * dho

        # Gradient for c in h = ho * tanh(c), note we're adding dc_next here
        dc = ho * dh_next * self.tanh.backward(c)
        dc = dc + dc_next

        # Gradient for hf in c = hf * c_old + hi * hc
        dhf = c_old * dc
        dhf = self.sigmoidf.backward(hf) * dhf

        # Gradient for hi in c = hf * c_old + hi * hc
        dhi = hc * dc
        dhi = self.sigmoidi.backward(hi) * dhi

        # Gradient for hc in c = hf * c_old + hi * hc
        dhc = hi * dc
        dhc = self.tanhc.backward(hc) * dhc

        # Gate gradients, just a normal fully connected layer gradient
        self.Wf.grad += hx.T @ dhf
        self.bf.grad += dhf.sum(axis=0)
        dxf = dhf @ self.Wf.data.T

        self.Wi.grad += hx.T @ dhi
        self.bi.grad += dhi.sum(axis=0)
        dxi = dhi @ self.Wi.data.T

        self.Wo.grad += hx.T @ dho
        self.bo.grad += dho.sum(axis=0)
        dxo = dho @ self.Wo.data.T

        self.Wc.grad += hx.T @ dhc
        self.bc.grad += dhc.sum(axis=0)
        dxc = dhc @ self.Wc.data.T

        # As x was used in multiple gates, the gradient must be accumulated here
        dx = dxo + dxc + dxi + dxf
        # Split the concatenated X, so that we get our gradient of h_old
        dh_next = dx[:, :self.hidden_size]
        # Gradient for c_old in c = hf * c_old + hi * hc
        dc_next = hf * dc

        return dh_next, dc_next

    def backward(self, delta):
        # https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/
        # https://gist.github.com/karpathy/d4dee566867f8291f086

        dh_next = delta
        dc_next = np.zeros_like(dh_next)
        for t in range(len(self.cache['hx']) - 1, 0, -1):
            dh_next, dc_next = self.backward_step(dh_next, dc_next, t)


def lstm_backward(prob, y_train, d_next, cache):
    # Unpack the cache variable to get the intermediate variables used in forward step
    # ... = cache
    dh_next, dc_next = d_next

    # Softmax loss gradient
    dy = prob.copy()
    dy[1, y_train] -= 1.

    # Hidden to output gradient
    dWy = h.T @ dy
    dby = dy
    # Note we're adding dh_next here
    dh = dy @ Wy.T + dh_next

    # Gradient for ho in h = ho * tanh(c)
    dho = tanh(c) * dh
    dho = dsigmoid(ho) * dho

    # Gradient for c in h = ho * tanh(c), note we're adding dc_next here
    dc = ho * dh * dtanh(c)
    dc = dc + dc_next

    # Gradient for hf in c = hf * c_old + hi * hc
    dhf = c_old * dc
    dhf = dsigmoid(hf) * dhf

    # Gradient for hi in c = hf * c_old + hi * hc
    dhi = hc * dc
    dhi = dsigmoid(hi) * dhi

    # Gradient for hc in c = hf * c_old + hi * hc
    dhc = hi * dc
    dhc = dtanh(hc) * dhc

    # Gate gradients, just a normal fully connected layer gradient
    dWf = X.T @ dhf
    dbf = dhf
    dXf = dhf @ Wf.T

    dWi = X.T @ dhi
    dbi = dhi
    dXi = dhi @ Wi.T

    dWo = X.T @ dho
    dbo = dho
    dXo = dho @ Wo.T

    dWc = X.T @ dhc
    dbc = dhc
    dXc = dhc @ Wc.T

    # As X was used in multiple gates, the gradient must be accumulated here
    dX = dXo + dXc + dXi + dXf
    # Split the concatenated X, so that we get our gradient of h_old
    dh_next = dX[:, :H]
    # Gradient for c_old in c = hf * c_old + hi * hc
    dc_next = hf * dc

    grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
    state = (dh_next, dc_next)

    return grad, state
