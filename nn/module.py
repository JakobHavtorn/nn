from collections import OrderedDict

import IPython
import numpy as np

from .parameter import Parameter


class Module():
    """Base class for all neural network modules such as activations and transformations.
    """

    def __init__(self, *args, **kwargs):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self.training = True

    def reset_cache(self):
        """Reset the cache of the module
        """
        self.cache = dict()

    def train(self, mode=True):
        """Recursively sets the traiing mode to `mode` for all submodules.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)

    def eval(self):
        """Recursively sets the training mode to evaluation for all submodules.
        """
        self.train(mode=False)

    def summarize(self):
        """Print a model summary including all submodules.
        """
        name_col_width = max(len(n) for n, m in self.named_modules()) + 2
        module_col_width = max(len(str(m)) for n, m in self.named_modules())
        for n, m in self.named_modules():
            if m is not self:
                print(n.ljust(name_col_width) + str(m).ljust(module_col_width))
        n = 0
        for p in self.parameters():
            n += np.prod(p.shape)
        print("Total number of parameters: " + str(n))

    def add_module(self, name, module):
        """Adds a child module to the current module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                type(module)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def named_modules(self, memo=None, prefix=''):
        """Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def children(self):
        """Returns an iterator over immediate children modules.
        """
        for _, module in self.named_children():
            yield module

    def named_parameters(self, memo=None, prefix=''):
        """Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        """
        if memo is None:
            memo = set()
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def parameters(self):
        """Returns an iterator over module parameters.
        """
        for _, param in self.named_parameters():
            yield param

    def _all_buffers(self, memo=None):
        """Returns an iterator over module buffers.
        """
        if memo is None:
            memo = set()
        for _, b in self._buffers.items():
            if b is not None and b not in memo:
                memo.add(b)
                yield b
        for module in self.children():
            for b in module._all_buffers(memo):
                yield b

    def register_buffer(self, name, tensor):
        """Adds a persistent buffer to the module.
        """
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif tensor is not None and not isinstance(tensor, np.ndarray):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(np.ndarray or None required)"
                            .format(type(tensor), name))
        else:
            self._buffers[name] = tensor

    def register_parameter(self, name, param):
        """Adds a parameter to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(type(param), name))
        else:
            self._parameters[name] = param

    def __getattr__(self, name):
        """Returns model parameter attributes by finding them in the _parameters dictionary attribute
        """ 
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        """Sets model parameter attributes by setting them in the _parameters dictionary attribute
        """

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]
        # Parameters
        parameters = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if parameters is None:
                raise AttributeError("cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif parameters is not None and name in parameters:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(type(value), name))
            self.register_parameter(name, value)
        else:  # Modules
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError("cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(nn.Module or None expected)"
                                    .format(type(value), name))
                modules[name] = value
            else:  # Buffers
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, np.ndarray):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(np.ndarray or None expected)"
                                        .format(type(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        """Deletes a model parameter attributes by deleting it in the _parameters dictionary attribute
        """
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)
    
    def reset_parameters(self):
        raise NotImplementedError()

    def forward(self, input):
        raise NotImplementedError()

    def backward(self, delta_in):
        raise NotImplementedError()
