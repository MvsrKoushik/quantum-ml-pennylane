import numpy as np
import pytest
from quantum_ml import angle_encode


def test_angle_encoding_range_and_constant_column():
    result = angle_encode([[1, 4], [3, 4]])
    assert np.all((0 <= result) & (result <= np.pi))
    assert np.all(result[:, 1] == 0)


def test_requires_matrix():
    with pytest.raises(ValueError):
        angle_encode([1, 2])

