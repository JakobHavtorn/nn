import numpy as np


class Parameter(object):
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
        # Gradient could be implemented lazily but that requires overloading
        # the `+=` etc. operators

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        p = Parameter(self.data.copy())
        p.grad = self.grad.copy()
        return p

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

    @staticmethod
    def get_data_array(other):
        if isinstance(other, Parameter):
            return other.data
        elif isinstance(other, (np.ndarray, float, int)):
            return other
        else:
            raise TypeError(f'Unknown type of other {type(other)}')

    def __getitem__(self, *args, **kwargs):
        return np.ndarray.__getitem__(self.data, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return np.ndarray.__setitem__(self.data, *args, **kwargs)

    def __add__(self, other):
        data_array = self.get_data_array(other)
        return np.add(self.data, data_array)

    def __sub__(self, other):
        data_array = self.get_data_array(other)
        return np.subtract(self.data, data_array)

    def __mul__(self, other):
        data_array = self.get_data_array(other)
        return np.multiply(self.data, data_array)

    def __truediv__(self, other):
        data_array = self.get_data_array(other)
        return np.divide(self.data, data_array)

    def __pow__(self, other):
        data_array = self.get_data_array(other)
        return np.power(self.data, data_array)

    def __radd__(self, other):
        data_array = self.get_data_array(other)
        return self.__add__(data_array)

    def __rsub__(self, other):
        data_array = self.get_data_array(other)
        return self.__sub__(data_array)

    def __rmul__(self, other):
        data_array = self.get_data_array(other)
        return self.__mul__(data_array)

    def __rdiv__(self, other):
        data_array = self.get_data_array(other)
        return np.divide(data_array, self.data)

    def __rpow__(self, other):
        data_array = self.get_data_array(other)
        return np.power(data_array, self.data)

    def __iadd__(self, other):
        data_array = self.get_data_array(other)
        return np.add(self.data, data_array, out=self.data)

    def __isub__(self, other):
        data_array = self.get_data_array(other)
        return np.subtract(self.data, data_array, out=self.data)

    def __imul__(self, other):
        data_array = self.get_data_array(other)
        return np.multiply(self.data, data_array, out=self.data)

    def __idiv__(self, other):
        data_array = self.get_data_array(other)
        return np.divide(self.data, data_array, out=self.data)

    def __ipow__(self, other):
        data_array = self.get_data_array(other)
        return np.power(self.data, data_array, out=self.data)

"""
Overview of Magic Methods
    Binary Operators
        Operator	Method                              Implemented
        +	        __add__(self, other)                [X]
        -	        __sub__(self, other)                [X]
        *	        __mul__(self, other)                [X]
        //	        __floordiv__(self, other)           [ ]
        /	        __truediv__(self, other)            [X]
        %	        __mod__(self, other)                [ ]
        **	        __pow__(self, other[, modulo])      [X]
        <<	        __lshift__(self, other)             [ ]
        >>	        __rshift__(self, other)             [ ]
        &	        __and__(self, other)                [ ]
        ^	        __xor__(self, other)                [ ]
        |	        __or__(self, other                  [ ]

    Extended Assignments
        Operator	Method                              Implemented
        +=	         __iadd__(self, other)              [X]
        -=	         __isub__(self, other)              [X]
        *=	         __imul__(self, other)              [X]
        /=	         __idiv__(self, other)              [X]
        //=	         __ifloordiv__(self, other)         [ ]
        %=	         __imod__(self, other)              [ ]
        **=	         __ipow__(self, other[, modulo])    [X]
        <<=	         __ilshift__(self, other)           [ ]
        >>=	         __irshift__(self, other)           [ ]
        &=	         __iand__(self, other)              [ ]
        ^=	         __ixor__(self, other)              [ ]
        |=	         __ior__(self, other)               [ ]

    Unary Operators
        Operator	Method                              Implemented
        -	        __neg__(self)                       [ ]
        +   	    __pos__(self)                       [ ]
        abs()	    __abs__(self)                       [ ]
        ~	        __invert__(self)                    [ ]
        complex()   __complex__(self)                   [ ]
        int()	    __int__(self)                       [ ]
        long()	    __long__(self)                      [ ]
        float()	    __float__(self)                     [ ]
        oct()	    __oct__(self)                       [ ]
        hex()	    __hex__(self                        [ ]

    Comparison Operators
        Operator	Method                              Implemented
        <	        __lt__(self, other)                 [ ]
        <=	        __le__(self, other)                 [ ]
        ==	        __eq__(self, other)                 [ ]
        !=	        __ne__(self, other)                 [ ]
        >=	        __ge__(self, other)                 [ ]
        >	        __gt__(self, other)                 [ ]
"""


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