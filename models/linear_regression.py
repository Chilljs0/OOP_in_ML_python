import numpy as np

class LinearRegression:
    """
    Linear Regression using normal equation.

    Attributes:
        coefficients (np.ndarray): The weights for the model.
        intercept (float): The bias for the linear model.
        verbose (bool): True prints detailed information during fitting and prediction.
    """

    def __init__(self, verbose=False):
        """
        Initializes the LinearRegression model.

        Args:
            verbose (bool): True enables verbose output for debugging.
        """
        self.coefficients = None
        self.intercept = None
        self.verbose = verbose

    def _log(self, message):
        """
        Helper method for logging messages.

        Args:
            message (str).
        """
        if self.verbose:
            print(message)

    def fit(self, X, y):
        """

        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).
            y (np.ndarray): The target values, shape (n_samples,).

        Raises:
            ValueError: If X or y is not a 2D or 1D array.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")

        X_with_intercept = np.c_[np.ones(X.shape[0]), X]

        self._log("Fitting the model...")
      
        """Normal equation"""
      
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self._log(f"Model fitted. Intercept: {self.intercept}, Coefficients: {self.coefficients}")

    def predict(self, X):
        """
        Makes predictions using the trained Linear Regression model.

        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted values, shape (n_samples,).

        Raises:
            ValueError: If X is not a 2D numpy array.
        """
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("X must be a 2D numpy array.")

        self._log("Making predictions...")
        predictions = self.intercept + X @ self.coefficients
        self._log(f"Predictions: {predictions}")
        return predictions

    def R2(self, X, y):
        """
        Calculates the R-squared value to evaluate performance.

        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).
            y (np.ndarray): The true target values, shape (n_samples,).

        Returns:
            float: The R-squared value.

        Raises:
            ValueError: If X or y is not a 2D or 1D array.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")

        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        self._log(f"R-squared: {r_squared}")
        return r_squared
