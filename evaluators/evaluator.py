class Evaluator:
    def __init__(self, evaluation_metric):
        """Initialize the evaluator
        """
        self.evaluation_metric_name = evaluation_metric

    @property
    def evaluation_metric(self):
        return getattr(self, self.evaluation_metric_name)

    def update(self, predictions=None, labels=None, loss=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
