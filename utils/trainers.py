import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import progressbar
import IPython

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

        lr_scheduler: optim.LRScheduler
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
        checkpoint_dir: str
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
        self.best_val_metric = kwargs.pop('epochs_no_improvement', -np.inf)
        self.lr_scheduler = kwargs.pop('lr_scheduler', None)
        self.max_epochs = kwargs.pop('max_epochs', 10)
        self.max_epochs_no_improvement = kwargs.pop('max_epochs_no_improvement', self.max_epochs)
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', None)

        self.num_classes = np.unique(self.train_loader.dataset.train_labels).shape[0]

        # Throw an error if there are extra keyword arguments
        if kwargs:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

    def reset(self):
        self.epoch = 0
        self.epochs_no_improvement = 0
        self.best_val_metric = -np.inf

    def step(self, pbar):
        raise NotImplementedError

    def validate(self, pbar):
        raise NotImplementedError

    def train(self):
        """Run optimization to train the model.
        """
        while self.epoch < self.max_epochs and self.epochs_no_improvement < self.max_epochs_no_improvement:

            print(f'Epoch {self.epoch:3d} | lr={self.optimizer.lr:4.5f}')

            # Progressbar
            widgets = [progressbar.FormatLabel(f'Epoch {self.epoch:3d} | Batch '),
                       progressbar.SimpleProgress(), ' | ',
                       progressbar.Percentage(), ' | ',
                       progressbar.FormatLabel(f'Loss N/A'), ' | ',
                       progressbar.Timer(), ' | ',
                       progressbar.ETA()]
            pbar_train = progressbar.ProgressBar(widgets=widgets)
            pbar_val = progressbar.ProgressBar(widgets=widgets)

            # Execute training on training set
            self.step(pbar_train)

            # Validate model on validation set
            self.validate(pbar_val)

            # Learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(f'Epoch {self.epoch:3d} | Loss (T/V) {self.train_evaluator.loss:5.4f} / {self.val_evaluator.loss:5.4f} | ' \
                  f'{self.train_evaluator.evaluation_metric_name.capitalize()} (T/V) {self.train_evaluator.evaluation_metric:5.4f} / {self.val_evaluator.evaluation_metric:5.4f}')

            # Keep track of the best model
            if self.val_evaluator.evaluation_metric > self.best_val_metric:
                self.best_val_metric = self.val_evaluator.evaluation_metric
                self._save_checkpoint()
                self.epochs_no_improvement = 0
            else:
                self.epochs_no_improvement += 1

            # Update plots
            self._update_plots()

            self.epoch += 1

    def _update_plots(self):
        for name, evaluator in zip(['train', 'val'], [self.train_evaluator, self.val_evaluator]):
            # Batch level
            for k in evaluator.history.keys():
                fig, ax = plt.subplots(figsize=(16, 9))
                evaluator.history[k].plot(ax=ax)
                ax.set_xlabel('Batch')
                ax.set_ylabel(k)
                fig.savefig(os.path.join(self.checkpoint_dir, name + '_' + k + '.pdf'), bbox_inches='tight')
                plt.close(fig)
            # # Epoch level
            # for k in evaluator.history.keys():
            #     fig, ax = plt.subplots(figsize=(16, 9))
            #     evaluator.history[k].plot(ax=ax)
            #     ax.set_xlabel('Batch')
            #     ax.set_ylabel(k)
            #     fig.savefig(os.path.join(self.checkpoint_dir, name + '_' + k + '.pdf'), bbox_inches='tight')
            #     plt.close(fig)

    def _save_checkpoint(self):
        if self.checkpoint_dir is None:
            return
        checkpoint = {
            'model': self.model,
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'train_evaluator': self.train_evaluator,
            'val_evaluator': self.val_evaluator,
            'epoch': self.epoch,
            'epochs_no_improvement': self.epochs_no_improvement,
            'num_classes': self.num_classes,
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_{id(self)}.pkl')
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

    def step(self, pbar):
        """Make a single gradient update.

        This is called by train()
        """
        self.model.train()
        self.train_evaluator.reset()
        for data, targets in pbar(self.train_loader):
            data, targets = data.numpy(), targets.numpy()
            
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

            pbar.widgets[5] = progressbar.FormatLabel(f'Loss (E/B) {self.train_evaluator.loss:4.2f} / {loss.mean():4.2f}')

    def validate(self, pbar):
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
        for data, targets in pbar(self.val_loader):
            data, targets = data.numpy(), targets.numpy()
            targets = onehot(targets, self.num_classes)
            scores = self.model.forward(data)
            loss = self.loss.forward(scores, targets)
            self.val_evaluator.update(scores, targets, loss)

            pbar.widgets[5] = progressbar.FormatLabel(f'Loss (E/B) {self.val_evaluator.loss:4.2f} / {loss.mean():4.2f}')
