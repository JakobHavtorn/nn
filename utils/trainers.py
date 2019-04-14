import pickle

import IPython
import numpy as np
import torch
import progressbar

from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms

from context import nn, optim
from .utils import onehot


class Trainer():
    """A Trainer encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, first construct a Solver instance, passing the
    model, dataset, optimizer, loss and various options (number of epochs, checkpoints,
     etc.) to the constructor. Then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, the model will contain the parameters
    that performed best on the validation set over the course of training.
    
    In addition, the instance variable solver.train_loss_history will contain a list
    of all losses encountered during training and solver.train_acc_history will contain
    the associated classification accuracies.
    """
    def __init__(self, model, optimizer, loss, train_loader, val_loader, train_evaluator, val_evaluator, **kwargs):
        """Construct a new Solver instance.

        Parameters
        ----------
        model: nn.Module
            The model to train
        optimizer: optim.Optimizer
            The optimizer from optim.py to use.
        loss: nn.Module
            The loss criterion to optimize.
        train_loader: torch.utils.data.DataLoader
            torch DataLoader constructed on the wanted training set.
        val_loader: torch.utils.data.DataLoader
            torch DataLoader constructed on the corresponding validation set.
        train_evaluator: eval.Evaluator

        val_evaluator: eval.Evaluator

        lr_decay: optim.LRScheduler
            A scheduler for learning rate decay.
        max_epochs: int
            The number of epochs to run for during training.
        max_epochs_no_improvement: int
            The number of epochs without improvement on the validation set to tolerate before ending training.
            Defaults to None.
        print_every: int
            Training losses will be printed every print_every iterations.
        verbose: bool
            If set to false then no output will be printed
          during training.
        checkpoint_name: str
            If not None, then save model checkpoints here every epoch.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss = loss
        self.train_evaluator = train_evaluator
        self.val_evaluator = val_evaluator

        # Unpack keyword arguments
        self.epoch = kwargs.pop('epoch', 0)
        self.epochs_no_improvement = kwargs.pop('epochs_no_improvement', 0)
        self.lr_decay = kwargs.pop('lr_decay', None)
        self.max_epochs = kwargs.pop('max_epochs', 10)
        self.max_epochs_no_improvement = kwargs.pop('max_epochs_no_improvement', None)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        
        self.num_classes = np.unique(self.train_loader.dataset.train_labels).shape[0]

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

    def _save_checkpoint(self):
        if self.checkpoint_name is None: 
            return
        checkpoint = {
          'model': self.model,
          'train_loader': self.train_loader,
          'val_loader': self.val_loader,
          'optimizer': self.optimizer,
          'lr_decay': self.lr_decay,
          'train_evaluator': self.train_evaluator,
          'val_evaluator': self.val_evaluator,
          'epoch': self.epoch,
          'epochs_no_improvement': self.epochs_no_improvement
          'num_classes': self.num_classes,
        }
        filename = '{:s}_epoch_{:d}.pkl'.format(self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "{:s}"'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)


class ClassificationTrainer(Trainer):
    """A Trainer encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, first construct a Solver instance, passing the
    model, dataset, optimizer, loss and various options (number of epochs, checkpoints,
     etc.) to the constructor. Then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, the model will contain the parameters
    that performed best on the validation set over the course of training.

    In addition, the instance variable solver.train_loss_history will contain a list
    of all losses encountered during training and solver.train_acc_history will contain
    the associated classification accuracies.
    """
    def __init__(self, model, optimizer, loss, train_loader, val_loader, train_evaluator, val_evaluator, **kwargs):
        super().__init__(model, optimizer, loss, train_loader, val_loader, train_evaluator, val_evaluator, **kwargs)

    def _step(self, data, targets):
        """Make a single gradient update.

        This is called by train()
        """
        targets = onehot(targets, self.num_classes)
        # Forward pass, compute loss
        scores = self.model.forward(data)
        loss = self.loss.forward(scores, targets)
        # Backward pass, compute gradient
        self.optimizer.zero_grad()
        dout = self.loss.backward(scores, targets)
        self.model.backward(dout)
        # Take step with optimizer
        self.optimizer.step()
        # Update evaluator
        self.train_evaluator.update(scores, targets, loss)
        return loss

    def validate(self):
        """Validates the model on the validation set.

        Returns
        -------
        accuracy: float
            Scalar giving the fraction of instances that were correctly classified by the model.
        loss: float
            Computed average loss on the validation set.
        """
        self.model.eval()
        self.val_evaluator.reset()
        for data, targets in self.val_loader:
            data, targets = data.numpy(), targets.numpy()
            targets_onehot = onehot(targets, self.num_classes)
            scores = self.model.forward(data)
            loss = self.loss.forward(scores, targets_onehot)
            self.val_evaluator.update(scores, targets, loss)

    def train(self):
        """Run optimization to train the model.
        """
        self.epochs_no_improvement, self.best_val_acc = 0, 0
        while self.epoch < self.max_epochs and (self.max_epochs_no_improvement is None or
              self.max_epochs_no_improvement < self.epochs_no_improvement):
            # Progressbar
            widgets = [progressbar.FormatLabel(f'Epoch {epoch:3d} | Batch '),
                   progressbar.SimpleProgress(), ' | ',
                   progressbar.Percentage(), ' | ',
                   progressbar.FormatLabel(f'Loss N/A'), ' | ',
                   progressbar.Timer(), ' | ',
                   progressbar.ETA()]
            pbar = progressbar.ProgressBar(widgets=widgets)

            # Execute training on training set
            self.model.train()
            self.train_evaluator.reset()
            for data, targets in pbar(self.train_loader):
                data, targets = data.numpy(), targets.numpy()
                loss = self._step(data, targets)
                pbar.widgets[5] = progressbar.FormatLabel(f'Loss (E/B) {self.train_evaluator.loss:4.2f} / {loss:4.2f}')

            # Validate model on validation set
            self.validate()

            # Learning rate decay
            if self.lr_decay is not None:
                self.lr_decay.step()

            # Keep track of the best model
            if self.val_evaluator.accuracy > self.best_val_acc:
                self.best_val_acc = self.val_evaluator.accuracy
                self._save_checkpoint()
     
            # Print result of an epoch
            # print('(Epoch {:d} / {:d}) TA: {:3.2f}% | TL {:3.2f} | VA {:3.2f}% | VL {:3.2f}'.format(
            #             self.epoch+1, self.max_epochs, avg_train_acc, avg_train_loss,
            #             self.val_acc_history[-1], self.val_loss_history[-1]))
