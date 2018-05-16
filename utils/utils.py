import numpy as np


def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes), dtype=bool)
    for row, col in enumerate(t):
        out[row, col] = True
    return out
