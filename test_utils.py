import numpy as np
from linear_regression.utils import mean_squared_error, r2_score

def test_mse():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    assert np.isclose(mean_squared_error(y_true, y_pred), 0.375)

def test_r2_score():
    y_true = [1, 2, 3]
    y_pred = [1.1, 1.9, 3.1]
    assert np.isclose(r2_score(y_true, y_pred), 0.98, atol=0.01)
