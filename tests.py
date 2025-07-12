import numpy as np
import pytest
from linear_regression.models import LinearRegressionNormal, LinearRegressionGD

@pytest.fixture
def sample_data():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    return X, y

def test_normal_equation(sample_data):
    X, y = sample_data
    model = LinearRegressionNormal()
    model.fit(X, y)
    pred = model.predict([[5]])
    assert np.isclose(pred, 10.0, atol=1e-3)

def test_gradient_descent(sample_data):
    X, y = sample_data
    model = LinearRegressionGD(learning_rate=0.01, max_iters=10000)
    model.fit(X, y)
    pred = model.predict([[5]])
    assert np.isclose(pred, 10.0, atol=1e-3)
    assert len(model.loss_history) > 0

def test_different_input_formats():
    model = LinearRegressionNormal()
    X = [[1], [2], [3]]
    y = [2, 4, 6]
    model.fit(X, y)
    
    # Test different prediction formats
    assert np.isclose(model.predict([[4]]), 8)
    assert np.isclose(model.predict([4]), 8)
    assert np.isclose(model.predict(np.array([4])), 8)
    assert np.isclose(model.predict(np.array([[4]])), 8)
