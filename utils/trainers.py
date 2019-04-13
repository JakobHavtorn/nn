import os
import pickle as pickle

import IPython
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from context import nn, optim
from torchvision import datasets, transforms

from .progress import ProgressBar
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
    def __init__(self, model, train_loader, val_loader, optimizer, loss, **kwargs):
        """Construct a new Solver instance.

        Parameters
        ----------
        model: Module
            The model to train
        train_loader: torch.utils.data.DataLoader
            torch DataLoader constructed on the wanted training set.
        val_loader: torch.utils.data.DataLoader
            torch DataLoader constructed on the corresponding validation set.
        optimizer: Optimizer
            The optimizer from optim.py to use.
        loss: Module
            The loss criterion to optimize.
        lr_decay: float
            A scalar for learning rate decay; after each epoch the learning rate 
            is multiplied by this value.
        num_epochs: int
            The number of epochs to run for during training.
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

        # Unpack keyword arguments
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_classes = np.unique(self.train_loader.dataset.train_labels).shape[0]
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)


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
    def __init__(self, model, train_loader, val_loader, optimizer, loss, **kwargs):
        super().__init__(model, train_loader, val_loader, optimizer, loss, **kwargs)
        self._reset()

    def _reset(self):
        """Set up some book-keeping variables for optimization. 
        
        Don't call this manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.train_acc_history = []
        self.train_loss_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def _step(self, data, targets):
        """Make a single gradient update. 
        
        This is called by train()
        """
        # Forward pass, compute loss
        scores = self.model.forward(data)
        loss = self.loss.forward(scores, targets)
        # Backward pass, compute gradient
        self.optimizer.zero_grad()
        dout = self.loss.backward(scores, targets)
        self.model.backward(dout)
        # Take step with optimizer
        self.optimizer.step()
        # Save loss history
        predictions = np.argmax(scores, axis=1)
        accuracy = np.sum(predictions == np.where(targets)[1]) / targets.shape[0] * 100
        self.train_acc_history.append(accuracy)
        self.train_loss_history.append(loss)

    def _save_checkpoint(self):
        if self.checkpoint_name is None: 
            return
        checkpoint = {
          'model': self.model,
          'train_loader': self.train_loader,
          'val_loader': self.val_loader,
          'optimizer': self.optimizer,
          'lr_decay': self.lr_decay,
          'num_classes': self.num_classes,
          'epoch': self.epoch,
          'train_loss_history': self.train_loss_history,
          'train_acc_history': self.train_acc_history,
          'val_loss_history': self.val_loss_history,
          'val_acc_history': self.val_acc_history,
        }
        filename = '{:s}_epoch_{:d}.pkl'.format(self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "{:s}"'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def validate_model(self):
        """Validates the model on the validation set.

        Returns
        -------
        accuracy: float
            Scalar giving the fraction of instances that were correctly classified by the model.
        loss: float
            Computed average loss on the validation set.
        """
        loss = 0
        correct = 0
        n = 0
        for data, targets in self.val_loader:
            data, targets = data.numpy(), targets.numpy()
            n += targets.shape[0]
            targets_onehot = onehot(targets, self.num_classes)
            scores = self.model.forward(data)
            loss += self.loss.forward(scores, targets_onehot)
            predictions = np.argmax(scores, axis=1)
            correct += np.sum(predictions == targets)
        loss /= len(self.val_loader)
        accuracy = correct / n * 100
        return accuracy, loss

    def train(self):
        """Run optimization to train the model.
        """
        self.batch_size = self.train_loader.batch_size
        num_train_examples = self.train_loader.dataset.train_labels.shape[0]
        self.batches_per_epoch = max(num_train_examples // self.batch_size, 1)
        if self.verbose:
            print('Training for {:d} epochs with {:d} batches per epoch...'.format(self.num_epochs, self.batches_per_epoch))
        for epoch in range(self.num_epochs):
            # Start monitoring progress
            self.epoch = epoch
            if self.verbose:
                title = "(Epoch {:d} / {:d})".format(self.epoch+1, self.num_epochs)
                self.progress = ProgressBar(title=title, end_value=self.batches_per_epoch, keep_after_done=False)
                self.progress.start()
            # Execute training and training set
            for batch, (data, targets) in enumerate(self.train_loader):
                data, targets = data.numpy(), targets.numpy()
                targets_onehot = onehot(targets, self.num_classes)
                self._step(data, targets_onehot)
                if self.verbose:
                    self.progress.progress(batch)
            # Decay learning rate
            self.optimizer.lr *= self.lr_decay
            # Validate
            self.model.eval()
            val_acc, val_loss = self.validate_model()
            self.model.train()
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            self._save_checkpoint()
            # Keep track of the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {}
                for k, p in self.model.named_parameters():
                    self.best_params[k] = p.data.copy()
            # Print result of epoch
            if self.verbose:
                self.progress.end()
                n_examples = min(len(self.train_acc_history), 100)
                avg_train_acc = np.mean(self.train_acc_history[-n_examples:])
                avg_train_loss = np.mean(self.train_loss_history[-n_examples:])
                print('(Epoch {:d} / {:d}) TA: {:3.2f}% | TL {:3.2f} | VA {:3.2f}% | VL {:3.2f}'.format(
                           self.epoch+1, self.num_epochs, avg_train_acc, avg_train_loss,
                           self.val_acc_history[-1], self.val_loss_history[-1]))

        # At the end of training swap the best params into the model
        for k, p in self.model.named_parameters():
            p = self.best_params[k]
