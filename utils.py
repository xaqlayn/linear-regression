import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error (MSE)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """Calculate R-squared (coefficient of determination)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
  
