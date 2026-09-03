import numpy as np


def angle_encode(features):
    values = np.asarray(features, dtype=float)
    if values.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    span[span == 0] = 1
    return np.pi * (values - minimum) / span

