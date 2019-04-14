class Evaluator:
    def __init__(self):
        """Initialize the evaluator
        """
        pass

    def update(self, predictions=None, labels=None, loss=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
