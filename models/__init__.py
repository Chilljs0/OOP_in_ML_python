"""
models package

Implementation Linear Regression and KNN.
"""
"""Imports models from current folder"""
from .linear_regression import LinearRegression
from .knn import KNN

"""import * from models"""
__all__ = ["LinearRegression", "KNN"]
