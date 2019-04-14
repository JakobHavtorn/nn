import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .evaluator import Evaluator


class MulticlassEvaluator(Evaluator):
    """
    Evaluator for multiclass classification.

    The confusion matrix may be replaced by the PyCM version (https://github.com/sepandhaghighi/pycm).
    This e.g. supports class weights and activation thresholds for computing the confusion matrix from
    class probabilities rather than labels. The only issue is how we accumulate the confusion matrices
    in this case since the classes don't support addition.
    """
    def __init__(self, n_classes=None, labels=None):
        """
        Initialize the MulticlassEvaluator object

        Args:
            n_classes (int): Number of classes
            labels (list): The labels for each class
        """
        super().__init__(self)
        if n_classes is not None:
            self._n_classes = n_classes
            self._labels = np.arange(0, n_classes, dtype=np.int)
        elif labels is not None:
            self._n_classes = len(labels)
            self._labels = labels
        self.reset()  # Setting all tracked metrics of the evaluator to default values.

    def update(self, predictions, labels, loss):
        """
        Update the tracked metrics: Confusion matrix, accuracy

        Args:
            predictions (list): List of predictions.
            labels (list): The labels corresponding to the predictions.
            loss (None or list): List of the loss for each example for each GPU.
        """
        # Update loss related values; remember to filter out infs and nans.
        # loss = np.concatenate(loss, axis=0)
        filter_naninf = np.invert(np.isinf(loss) + np.isnan(loss))
        loss_clean = loss[filter_naninf]
        self.loss_sum += np.sum(loss_clean)
        self.num_examples += len(loss_clean)
        self.loss = self.loss_sum / self.num_examples

        # Update confusion matrix
        # predictions = np.concatenate([pred for pred in predictions])
        labels = np.concatenate([lab.argmax(axis=1) for lab in labels])  # Decode one-hot encoding
        # Confusion matrix with model predictions in rows, true labels in columns
        self.cm += confusion_matrix(labels, predictions, labels=self._labels)
        self.tp = np.diag(self.cm)  # TPs are diagonal elements
        self.fp = self.cm.sum(axis=0) - self.tp  # FPs is sum of row minus true positives
        self.fn = self.cm.sum(axis=1) - self.tp  # FNs is sum of column minus true positives
        # TNs is the total count minus false positives and false negatives plus true positives (which are otherwise subtracted twice)
        self.tn = self.cm.sum() - np.array([self.cm[i, :].sum() + self.cm[:, i].sum() - self.cm[i, i] for i in range(self._n_classes)])

    @property
    def recalls(self):
        return self.tp / np.maximum(self.tp + self.fn, 1)

    @property
    def precisions(self):
        return self.tp / np.maximum(self.tp + self.fp, 1)

    @property
    def f1_scores(self):
        return 2 * self.tp / np.maximum(2 * self.tp + self.fn + self.fp, 1)

    @property
    def accuracies(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def recall(self):
        return self.recalls

    @property
    def precision(self):
        return self.precisions

    @property
    def f1_score(self):
        return self.f1_scores

    @property
    def accuracy(self):
        return self.accuracies

    def reset(self):
        """
        Reset the tracked metrics.
        """
        self.loss_sum = 0
        self.num_examples = 0
        self.loss = None
        self.tp = np.zeros(shape=self._n_classes)
        self.fp = np.zeros(shape=self._n_classes)
        self.fn = np.zeros(shape=self._n_classes)
        self.tn = np.zeros(shape=self._n_classes)
        self.cm = np.zeros((self._n_classes, self._n_classes))
