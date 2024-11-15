"""
encoder.py

Class for encoding categorical features using one-hot encoding.
"""

import pandas as pd

class CategoricalEncoder:
  
    def __init__(self, drop_first=True):
        """
        Initializes the CategoricalEncoder.

        Args:
            drop_first (bool): If True, drops the first category in each encoded column.
        """
        self.drop_first = drop_first

    def fit_transform(self, X):
        """
        Applies one-hot encoding to categorical columns.

        Args:
            X (pd.DataFrame): The input features DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with encoded columns.
        """
      
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        X_encoded = pd.get_dummies(X, drop_first=self.drop_first)
        return X_encoded
