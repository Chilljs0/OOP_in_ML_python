"""
evaluator.py

Evaluator class, which uses metrics from Metrics.py to evaluate the performance of regression models.
"""

from evaluators.metrics import mean_squared_error, mean_absolute_error, r_squared, mean_absolute_percentage_error

class Evaluator:
    """
    A class to evaluate regression models using different metrics.

    Attributes:
        model: The model to evaluate.
    """

    def __init__(self, model):
        """
        Args:
            model: The regression model to evaluate.
        """
        self.model = model

    def evaluate(self, X, y):
        """
        Evaluates the model using MSE, MAE,
        R-squared, and MAPE.

        Args:
            X (np.ndarray): The input features for evaluation, shape (n_samples, n_features).
            y (np.ndarray): The true target values, shape (n_samples,).

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """

        y_pred = self.model.predict(X)


        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r_squared(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)


        print("Evaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R²): {r2}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

        return {
            "MSE": mse,
            "MAE": mae,
            "R-squared": r2,
            "MAPE": mape
        }
