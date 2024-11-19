import numpy as np
import matplotlib.pyplot as plt

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
        Args:
            verbose (bool): True enables verbose output for debugging.
        """
        self.coefficients = None
        self.intercept = None
        self.verbose = verbose

    def _log(self, message):
        """
        Logging messages.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            print(message)

    def fit(self, x, y):
        """

        Args:
            x (np.ndarray): The input features, shape (n_samples, n_features).
            y (np.ndarray): The target values, shape (n_samples,).

        Raises:
            ValueError: If x or y is not a 2D or 1D array.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays.")
        if len(x.shape) != 2 or len(y.shape) != 1:
            raise ValueError("x must be a 2D array and y must be a 1D array.")

        x_with_intercept = np.c_[np.ones(x.shape[0]), x]

        self._log("Fitting the model...")
        beta = np.linalg.inv(x_with_intercept.T @ x_with_intercept) @ x_with_intercept.T @ y
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self._log(f"Model fitted. Intercept: {self.intercept}, Coefficients: {self.coefficients}")

    def predict(self, x):
        """
        Args:
            x (np.ndarray): The input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted values, shape (n_samples,).

        Raises:
            ValueError: If x is not a 2D numpy array.
        """
        if not isinstance(x, np.ndarray) or len(x.shape) != 2:
            raise ValueError("x must be a 2D numpy array.")

        self._log("Making predictions...")
        predictions = self.intercept + x @ self.coefficients
        # self._log(f"Predictions: {predictions}")
        return predictions

    def R2(self, x, y):
        """
        Args:
            x (np.ndarray): The input features, shape (n_samples, n_features).
            y (np.ndarray): The true target values, shape (n_samples,).

        Returns:
            float: The R-squared value.

        Raises:
            ValueError: If x or y is not a 2D or 1D numpy array.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays.")
        if len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError("x must be a 2D array and y must be a 1D array.")

        y_pred = self.predict(x)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        self._log(f"R-squared: {r_squared}")
        return r_squared


    def plot_predictions(self, x, y):
        """
        Plots the true values against the predicted values to visualize the performance
        of the regression model. The x-axis shows the predicted values, and the y-axis
        shows the true values.

        Args:
            x (np.ndarray): The input features, shape (n_samples, n_features).
            y (np.ndarray): The true target values, shape (n_samples,).

        Raises:
            ValueError: If x or y is not a 2D or 1D numpy array.
        """
        save_path = 'plots/LRplot.png'

        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays.")
        if len(x.shape) != 2 or len(y.shape) != 1:
            raise ValueError("x must be a 2D array and y must be a 1D array.")

        y_pred = self.predict(x)

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, y, color='blue', label='Data points', alpha=0.6)
        plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], color='red', linestyle='--', label='Perfect Fit Line')
        plt.xlabel("Predicted Values")
        plt.ylabel("True Values")
        plt.title("Linear Regression: True vs. Predicted Values")
        plt.legend()

        # Save the plot
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
