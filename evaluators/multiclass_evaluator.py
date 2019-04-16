import numpy as np

import pandas as pd

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
    def __init__(self, n_classes=None, labels=None, evaluation_metric='accuracy'):
        """
        Initialize the MulticlassEvaluator object

        Args:
            n_classes (int): Number of classes
            labels (list): The labels for each class
            evaluation_metric (str): The attribute to use as evaluation metric
        """
        super().__init__(evaluation_metric)
        if n_classes is not None:
            self._n_classes = n_classes
        if labels is not None:
            assert self._n_classes == len(labels), 'Must have as many labels as classes'
            self._labels = labels
        else:
            self._labels = np.arange(0, n_classes, dtype=np.int)
        self.batch = 0
        self._track_metrics = ('loss', 'accuracies', 'f1_scores', 'tprs', 'fprs', 'tnrs', 'fnrs', 'ppvs', 'fors', 'npvs', 'fdrs')
        self.history = pd.DataFrame(columns=('batch',) + self._track_metrics)
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
        loss_filter = np.invert(np.logical_or(np.isinf(loss), np.isnan(loss)))
        loss = loss[loss_filter]
        self.loss_sum += loss.sum()
        self.num_examples += loss.size
        loss = loss.mean()

        # Update confusion matrix
        # Confusion matrix with model predictions in rows, true labels in columns
        # Batch statistics for history
        cm = confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1), labels=self._labels)
        tps, fps, tns, fns = self.compute_tp_fp_tn_fn(cm)
        tprs, fprs, tnrs, fnrs = self.compute_tpr_fpr_tnr_fnr(tps, fps, tns, fns)
        ppvs, fors, npvs, fdrs = self.compute_ppv_for_npv_fdr(tps, fps, tns, fns)
        accuracies = self.compute_accuracies(tps, fps, tns, fns)
        f1_scores = self.compute_f1_scores(tps, fps, tns, fns)
        d = {'batch': self.batch}
        for v in self._track_metrics:
            d.update({v: eval(v).mean()})
        self.history = self.history.append(d, ignore_index=True)

        # accumulated statistics
        self.cm += cm
        self.tps, self.fps, self.tns, self.fns = self.compute_tp_fp_tn_fn(self.cm)
        self.tprs, self.fprs, self.tnrs, self.fnrs = self.compute_tpr_fpr_tnr_fnr(self.tps, self.fps, self.tns, self.fns)
        self.ppvs, self.fors, self.npvs, self.fdrs = self.compute_ppv_for_npv_fdr(self.tps, self.fps, self.tns, self.fns)
        self.accuracies = self.compute_accuracies(self.tps, self.fps, self.tns, self.fns)
        self.f1_scores = self.compute_f1_scores(self.tps, self.fps, self.tns, self.fns)

        # Bump batch counter
        self.batch += 1

    @staticmethod
    def compute_tp_fp_tn_fn(cm):
        tp = np.diag(cm)  # TPs are diagonal elements
        fp = cm.sum(axis=0) - tp  # FPs is sum of row minus true positives
        fn = cm.sum(axis=1) - tp  # FNs is sum of column minus true positives
        tn = cm.sum() - np.array([cm[i, :].sum() + cm[:, i].sum() - cm[i, i] for i in range(cm.shape[0])])  # total count minus false positives and false negatives plus true positives (which are otherwise subtracted twice)
        return tp, fp, tn, fn

    @staticmethod
    def compute_tpr_fpr_tnr_fnr(tp, fp, tn, fn):
        TPRs = tp / np.maximum(tp + fn, 1)  # True positive rate (recall)
        FPRs = fp / np.maximum(tn + fp, 1)  # False positive rate
        TNRs = tn / np.maximum(tn + fp, 1)  # True negative rate (specificity)
        FNRs = fn / np.maximum(tp + fn, 1)  # False negative rate
        return TPRs, FPRs, TNRs, FNRs

    @staticmethod
    def compute_ppv_for_npv_fdr(tp, fp, tn, fn):
        PPVs = tp / np.maximum(tp + fp, 1)  # Positive predictive value (precision)
        FORs = fn / np.maximum(tn + fn, 1)  # False omission rate
        NPVs = tn / np.maximum(tn + fn, 1)  # Negative predictive value
        FDRs = fp / np.maximum(tp + fp, 1)  # False discovery rate
        return PPVs, FORs, NPVs, FDRs

    @staticmethod
    def compute_accuracies(tp, fp, tn, fn):
        return (tp + tn) / np.maximum(tp + fp + tn + fn, 1)

    @staticmethod
    def compute_f1_scores(tp, fp, tn, fn):
        return 2 * tp / np.maximum(2 * tp + fn + fp, 1)

    @property
    def loss(self):
        return self.loss_sum / self.num_examples

    @property
    def tpr(self):
        return self.tprs.mean()

    @property
    def ppv(self):
        return self.ppvs.mean()

    @property
    def f1_score(self):
        return self.f1_scores.mean()

    @property
    def accuracy(self):
        return self.accuracies.mean()

    def reset(self):
        """
        Reset the tracked metrics.
        """
        self.loss_sum = 0
        self.num_examples = 0
        self.tps = np.zeros(shape=self._n_classes)
        self.fps = np.zeros(shape=self._n_classes)
        self.fns = np.zeros(shape=self._n_classes)
        self.tns = np.zeros(shape=self._n_classes)
        self.cm = np.zeros((self._n_classes, self._n_classes))
