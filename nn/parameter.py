import numpy as np


class Parameter(object):
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
        # Gradient could be implemented lazily but that requires overloading
        # the `+=` etc. operators
    
    def __repr__(self):
        s = "Parameter containing:\n"
        s += self.data.__repr__()
        s += "\n" + '(' + ', '.join([str(s) for s in self.data.shape]) + ')'
        if self.grad is not None:
            s += "\n\nand gradient\n\n"
            s += self.grad.__repr__()
            s += "\n" + '(' + ', '.join([str(s) for s in self.data.shape]) + ')'
        s += "\n"
        return s

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        p = Parameter(self.data.copy())
        p.grad = self.grad.copy()
        return p


class LazyArray(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = input_array
        # Finally, we must return the newly created object:
        return obj

    def __add__(self, data):
        return self.data + data

    def __sub__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    def __div__(self, other):
        raise NotImplementedError()

    def __pow__(self, other):
        raise NotImplementedError()

    def __radd__(self, data):
        raise NotImplementedError()

    def __rsub__(self, data):
        raise NotImplementedError()

    def __rmul__(self, data):
        raise NotImplementedError()

    def __rdiv__(self, data):
        raise NotImplementedError()

    def __rpow__(self, data):
        raise NotImplementedError()

    def __iadd__(self, data):
        raise NotImplementedError()

    def __isub__(self, data):
        raise NotImplementedError()

    def __imul__(self, data):
        raise NotImplementedError()

    def __idiv__(self, data):
        raise NotImplementedError()

    def __ipow__(self, data):
        raise NotImplementedError()