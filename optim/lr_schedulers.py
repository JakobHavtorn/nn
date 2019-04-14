import numpy as np


class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        self.optimizer.lr = self.get_lr()
        
        # for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
        #     param_group['lr'] = lr


class ExponentialDecay(LRScheduler):
    def __init__(self, rate=None, half_time=10, last_epoch=-1):
        """Exponential learning rate decay
        
        Set the learning rate of each parameter group to the initial lr decayed
        by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            rate (float): Multiplicative factor of learning rate decay.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        assert (half_time is not None) != (rate is not None), 'One and only one of half_time and rate must be set'
        if half_time is not None:
            self.rate = np.log(2) / half_time
        elif rate is not None:
            self.rate = rate

    def get_lr(self):
        return self.optimizer.lr * self.rate ** self.last_epoch


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing of learning rate
    
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where eta_max is set to the initial lr and
    T_cur is the number of epochs since the last restart in SGDR:

        eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(T_cur / {T_max} * π))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts` [1].
    Note that this only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    [1] Stochastic Gradient Descent with Warm Restarts: https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.eta_min + (self.optimizer.lr - self.eta_min) * (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
