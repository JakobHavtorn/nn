class Evaluator:
    def __init__(self, evaluation_metric):
        """Initialize the evaluator
        """
        self._evaluation_metric = evaluation_metric

    @property
    def evaluation_metric(self):
        return getattr(self, self._evaluation_metric)

    def update(self, predictions=None, labels=None, loss=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
