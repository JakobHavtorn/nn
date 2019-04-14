import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

from .evaluator import Evaluator


class BinaryEvaluator(Evaluator):
    def __init__(self):
        """
        Initialize a BinaryEvaluator.

        The BinaryEvaluator is an evaluator to be used for binary classification tasks which can be formulated in terms
        of a detection. For this purpose, the evaluator implements methods for finding an optimal decision boundary
        while keeping the TPR or the FPR fixed.
        """
        super().__init__(self)

    def update(self, probabilities, labels, loss):
        """
        Update the tracked metrics: Recall/FPR and loss.

        Args:
            probabilities (list): List of predicted probabilities for the positive class for each example.
            labels (list): The labels corresponding to the predictions, as one-hot-encoded vectors.
            loss (list): List of the loss for each example for each GPU.
        """
        # Update loss related values; remember to filter out infs and nans.
        # loss = np.array(loss)
        filter_naninf = np.invert(np.isinf(loss) + np.isnan(loss))
        loss_clean = loss[filter_naninf]
        self.loss_sum += np.sum(loss_clean)
        self.num_examples += len(loss_clean)
        self.loss = self.loss_sum / self.num_examples

        # Decode predictions and sparse labels for WER computation.
        self.probabilities = np.append(self.probabilities, np.concatenate(probabilities, axis=0))
        self.labels = np.append(self.labels, np.argmax(np.concatenate(labels, axis=0), axis=1))

    @staticmethod
    def _compute_recall(predictions, targets):
        """
        Computes tpr for given binary predictions and targets.
        Args:
            predictions (list of int): Binary predictions.
            targets (list of int): Binary targets.

        Returns:
            float: Recall.
        """
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        tpr = tp / max(tp + fn, 1)
        return tpr

    @staticmethod
    def _compute_fpr(predictions, targets):
        """
        Computes false positive rate.

        Args:
            predictions (list of int): The binary predictions.
            targets (list of int): The targets.

        Returns:
            float: False positive rate.
        """
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        fpr = fp / max(fp + tn, 1)
        return fpr

    @staticmethod
    def _compute_metrics(predictions, targets):
        """
        Computes key metrics for predictions and targets.

        Args:
            predictions (list of int): The binary predictions for the targets.
            targets (list of int): The targets for which predictions are provided.

        Returns:
            tuple: A tuple with some key metrics.
        """
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()

        tpr = BinaryEvaluator._compute_recall(predictions, targets)
        fpr = BinaryEvaluator._compute_fpr(predictions, targets)

        ppv = tp / max(tp + fp, 1)
        tnr = tn / max(tn + fp, 1)
        fnr = fn / max(fn + tp, 1)
        npv = tn / max(tn + fn, 1)

        return tpr, fpr, ppv, tnr, fnr, npv, tp, tn, fp, fn

    def evaluate_area(self, metric):
        """
        Evaluates area under the curve metrics. AUC or AUCPR

        Args:
            metric (str): Can be 'auc' or 'aucpr' to evaluate the respective metric.

        Returns:
            float: The metric for the currently recorded predictions.
        """
        if metric == 'auc':
            return roc_auc_score(self.labels, self.probabilities)
        elif metric == 'aucpr':
            precision, recall, _ = precision_recall_curve(self.labels, self.probabilities)
            return auc(recall, precision)

    def evaluate_fixed_rate(self, fixed_tpr=None, fixed_fpr=None):
        """
        Finds and evaluates a decision boundary at either fixed TPR or fixed FPR.

        One and only one of `fixed_tpr` and `fixed_fpr` must be given.

        Args:
            fixed_tpr (float): The fixed tpr to use when finding the decision boundary
            fixed_fpr (float): The fixed fpr to use when finding the decision boundary

        Returns:
            tuple of decision boundary and metrics at the fixed rate.
        """
        decision_boundary, _, _ = self.find_decision_boundary(fixed_tpr, fixed_fpr)
        tpr, fpr, ppv, tnr, fnr, npv, tp, tn, fp, fn = self.evaluate_decision_boundary(decision_boundary)

        return decision_boundary, tpr, fpr, ppv, tnr, fnr, npv, tp, tn, fp, fn

    def find_decision_boundary(self, fixed_tpr=None, fixed_fpr=None):
        """
        Finds a decision boundary at either fixed TPR or fixed FPR.

        One and only one of `fixed_tpr` and `fixed_fpr` must be given.

        Args:
            fixed_tpr (float): If specified, returns decision boundary at this true positive rate
            fixed_fpr (float): If specified, returns decision boundary at this false positive rate

        Returns:
            tuple: The decision boundary, the tpr, and the false positive rate at the specified rate.
        """
        if (fixed_tpr is None) == (fixed_fpr is None):
            return ValueError('Either `fixed_tpr` or `fixed_fpr` have to be specified, but not both.')

        fpr, tpr, thresholds = roc_curve(self.labels, self.probabilities)

        if fixed_tpr is not None:
            index_best = np.argmax(tpr >= fixed_tpr)
            tpr = fixed_tpr
            fpr = fpr[index_best]
            decision_boundary = thresholds[index_best]
        else:
            index_best = max(0, np.argmin(fpr <= fixed_fpr) - 1)
            tpr = tpr[index_best]
            fpr = fixed_fpr
            decision_boundary = thresholds[index_best]

        return decision_boundary, tpr, fpr

    def evaluate_decision_boundary(self, decision_boundary):
        """
        Evaluates metrics at the specified decision boundary.

        Args:
            decision_boundary (float): Decision boundary at which to evaluate the metrics.

        Returns:
            tuple: A tuple of metrics at the specified decision boundary.
        """
        predictions = (self.probabilities >= decision_boundary).astype(int)
        tpr, fpr, ppv, tnr, fnr, npv, tp, tn, fp, fn = self._compute_metrics(predictions, self.labels)

        return tpr, fpr, ppv, tnr, fnr, npv, tp, tn, fp, fn

    def reset(self):
        """
        Reset the tracked metrics.
        """
        self.loss_sum = 0
        self.num_examples = 0
        self.loss = None
        self.probabilities = np.array(())
        self.labels = np.array(())
