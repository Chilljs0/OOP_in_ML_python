import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors algorithm.

    Attributes:
        k (int): The number of neighbors.
        verbose (bool): If True, prints detailed information during fitting and prediction.
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels.
    """

    def __init__(self, k=5, verbose=False):
        """

        Args:
            k (int): The number of neighbors to consider. Default is 5.
            verbose (bool): True enables verbose output for debugging.
        """
        self.k = k
        self.verbose = verbose
        self.X_train = None
        self.y_train = None

    def _log(self, message):
        """
        Helper method for logging messages.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            print(message)

    def fit(self, X, y):
        """
        Stores the training data.

        Args:
            X (np.ndarray): The training data features, shape (n_samples, n_features).
            y (np.ndarray): The training data labels, shape (n_samples,).

        Raises:
            ValueError: If X or y is not a 2D or 1D array.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")

        self.X_train = X
        self.y_train = y
        self._log("Model fitted with training data.")

    def _compute_distance(self, x1, x2):
        """
        Computes the Euclidean distance between two points.

        Args:
            x1 (np.ndarray): Data point.
            x2 (np.ndarray): Data point.

        Returns:
            float: The Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Predicts the labels for input data.

        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted labels, shape (n_samples,).

        Raises:
            ValueError: If X is not a 2D numpy array.
        """
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("X must be a 2D numpy array.")

        self._log("Making predictions...")
        predictions = [self._predict_single_point(x) for x in X]
        return np.array(predictions)

    def _predict_single_point(self, x):
        """
        Predicts the label for a single data point using KNN.

        Args:
            x (np.ndarray): The input data point, shape (n_features,).

        Returns:
            int/float: The predicted label for the data point.
        """
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        self._log(f"Distances: {distances}")

        k_indices = np.argsort(distances)[:self.k]
        self._log(f"Indices of k nearest neighbors: {k_indices}")

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        self._log(f"Labels of k nearest neighbors: {k_nearest_labels}")

        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def acc(self, X, y):
        """
        Calculates the accuracy of the model for classification tasks.

        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).
            y (np.ndarray): The true labels, shape (n_samples,).

        Returns:
            float: The accuracy score.

        Raises:
            ValueError: If X or y is not a 2D or 1D array.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")

        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        self._log(f"Accuracy: {accuracy}")
        return accuracy
