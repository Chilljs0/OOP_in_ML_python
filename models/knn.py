import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KNN:
    """
    Attributes:
        k (int): The number of neighbors.
        verbose (bool): True prints detailed information during fitting and prediction.
        x_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels.
    """

    def __init__(self, k=3, verbose=False):
        """
        Args:
            k (int): The number of neighbors.
            verbose (bool): True enables verbose output for debugging.
        """
        self.k = k
        self.verbose = verbose
        self.x_train = None
        self.y_train = None

    def _log(self, message):
        """
        Args:
            message (str): The message to log.
        """
        if self.verbose:
            print(message)

    def fit(self, x, y):
        """
        Args:
            x (np.ndarray): The training data features, shape (n_samples, n_features).
            y (np.ndarray): The training data labels, shape (n_samples,).

        Raises:
            ValueError: If x or y is not a 2D or 1D array.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays.")
        if len(x.shape) != 2 or len(y.shape) != 1:
            raise ValueError("x must be a 2D array and y must be a 1D array.")

        self.x_train = x
        self.y_train = y
        self._log("Model fitted with training data.")

    def _compute_distance(self, x1, x2):
        """
        Args:
            x1 (np.ndarray): The first data point.
            x2 (np.ndarray): The second data point.

        Returns:
            float: The Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single_point(self, x):
        """
        Args:
            x (np.ndarray): The input data point, shape (n_features,).

        Returns:
            int/float: The predicted label for the data point.
        """
        distances = [self._compute_distance(x, x_train) for x_train in self.x_train]
        self._log(f"Distances: {distances}")

        k_indices = np.argsort(distances)[:self.k]
        self._log(f"Indices of k nearest neighbors: {k_indices}")

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        self._log(f"Labels of k nearest neighbors: {k_nearest_labels}")

        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def predict(self, x):
        """
        Args:
            x (np.ndarray): The input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted labels, shape (n_samples,).

        Raises:
            ValueError: If x is not a 2D numpy array.
        """
        if not isinstance(x, np.ndarray) or len(x.shape) != 2:
            raise ValueError("x must be a 2D numpy array.")

        self._log("Making predictions...")
        predictions = [self._predict_single_point(x_point) for x_point in x]
        return np.array(predictions)

    def plot_neighbors(self, x, y, test_point):
        """
        Args:
            x (np.ndarray): The training data features, shape (n_samples, 2).
            y (np.ndarray): The training data labels, shape (n_samples,).
            test_point (np.ndarray): The test point to visualize, shape (2,).

        Raises:
            ValueError: If x does not have 2 features.
        """
        if x.shape[1] != 2:
            raise ValueError("plot_neighbors only works for 2D feature data.")

        distances = [self._compute_distance(test_point, x_train) for x_train in x]
        k_indices = np.argsort(distances)[:self.k]

        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(x[y == label, 0], x[y == label, 1], label=f"Class {label}", alpha=0.6)

        plt.scatter(x[k_indices, 0], x[k_indices, 1], color='red', edgecolor='black', label='Neighbors', s=100, marker='o')

        plt.scatter(test_point[0], test_point[1], color='black', label='Test Point', s=100, marker='x')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"K-Nearest Neighbors (k={self.k})")
        plt.legend()
        plt.show()
