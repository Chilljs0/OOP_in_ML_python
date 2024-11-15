"""
scaler.py

Class for scaling numerical features using standardization. Standardized to have mean 0 and variance 1. (z = (x - u) / s where u is the mean of the training samples and s is the standard deviation of the training samples).
"""

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

class StandardScaler:

    def __init__(self):
        """
        Initializes the StandardScaler.
        """
        self.scaler = SklearnStandardScaler()

    def fit_transform(self, X):
        """
        Fits the scaler to the data and then transforms it.

        Args:
            X (np.ndarray or pd.DataFrame): The input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed features.
        """

        if not hasattr(X, "shape"):
            raise ValueError("X must be a numpy array or pandas DataFrame.")

        return self.scaler.fit_transform(X)
