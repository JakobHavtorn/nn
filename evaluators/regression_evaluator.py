import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

from .evaluator import Evaluator


class RegressionEvaluator(Evaluator):
    def __init__(self, evaluation_metric='pearson'):
        """
        Initialize a RegressionEvaluator.

        The RegressionEvaluator is an evaluator to be used for regression tasks.
        """
        super().__init__(self, evaluation_metric)

    def update(self, probs=None, labels=None, loss=None):
        """
        Update the tracked metrics: Recall/FPR and loss.

        Args:
            probs (list): List of predicted probabilities for the positive class for each example.
            labels (list): The labels corresponding to the predictions, as one-hot-encoded vectors.
            loss (list): List of the loss for each example for each GPU.
        """
        # Update loss related values; remember to filter out infs and nans.
        loss = np.array(loss)
        filter_naninf = np.invert(np.isinf(loss) + np.isnan(loss))
        example_loss_clean = loss[filter_naninf]
        self.loss_sum += np.sum(example_loss_clean)
        self.num_examples += len(example_loss_clean)
        self.loss = self.loss_sum / self.num_examples

        # # Decode predictions and sparse labels for WER computation.
        # self.probs = np.append(self.probs, np.concatenate(probs, axis=0))
        # self.labels = np.append(self.labels, np.argmax(np.concatenate(labels, axis=0), axis=1))

    def reset(self):
        """
        Reset the tracked metrics.
        """
        self.loss_sum = 0
        self.num_examples = 0
        self.loss = None
        self.probs = np.array(())
        self.labels = np.array(())