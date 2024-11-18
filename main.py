"""
main.py

This main file contains a custom pipeline with a
Linear Regression and KNN model, using data from a CSV file.
"""
import numpy as np
from utils import load_data
from preprocessors import CategoricalEncoder, StandardScaler
from models import LinearRegression, KNN
from evaluators import Evaluator

def main():
    # Load the data
    data_path = "data/Housing.csv"
    data = load_data(data_path, verbose=True)

    if data is None:
        print("Data loading failed. Exiting the program.")
        return

    # Separate features and target variable
    target_column = 'price'
    y = data[target_column].values
    x = data.drop(columns=[target_column])

    # Preprocess the data
    encoder = CategoricalEncoder()
    x_encoded = encoder.fit_transform(x)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_encoded)

    # Train and evaluate the Linear Regression model
    print("\n--- Linear Regression Model ---")
    linear_model = LinearRegression(verbose=True)
    linear_model.fit(x_scaled, y)

    # Evaluate the Linear Regression model
    linear_evaluator = Evaluator(linear_model)
    linear_metrics = linear_evaluator.evaluate(x_scaled, y)
    print("Linear Regression Evaluation Results:", linear_metrics)

    # Train and evaluate the KNN model
    print("\n--- K-Nearest Neighbors Model ---")
    knn_model = KNN(k=5, verbose=True)
    knn_model.fit(x_scaled, y)

    # Evaluate the KNN model
    knn_evaluator = Evaluator(knn_model)
    knn_metrics = knn_evaluator.evaluate(x_scaled, y)
    print("KNN Evaluation Results:", knn_metrics)

if __name__ == "__main__":
    main()
