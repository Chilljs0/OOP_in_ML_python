"""
metrics.py

Eevaluating the performance of the regression models.
"""

import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculates the MSE between the true and predicted values.

    Args:
        y_true (np.ndarray): The true target values, shape (n_samples,).
        y_pred (np.ndarray): The predicted values, shape (n_samples,).

    Returns:
        float: The mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Calculates the MAE between the true and predicted values.

    Args:
        y_true (np.ndarray): The true target values, shape (n_samples,).
        y_pred (np.ndarray): The predicted values, shape (n_samples,).

    Returns:
        float: The mean absolute error.
    """
    return np.mean(np.abs(y_true - y_pred))

def r_squared(y_true, y_pred):
    """
    Calculates the coefficient of determination.

    Args:
        y_true (np.ndarray): The true target values, shape (n_samples,).
        y_pred (np.ndarray): The predicted values, shape (n_samples,).

    Returns:
        float: The R-squared value.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the MAPE between the true and predicted values.

    Args:
        y_true (np.ndarray): The true target values, shape (n_samples,).
        y_pred (np.ndarray): The predicted values, shape (n_samples,).

    Returns:
        float: The mean absolute percentage error.
    """
    """Avoid division by 0"""
    epsilon = np.finfo(float).eps
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
