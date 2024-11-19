"""
main.py

Custom machine learning pipeline with Linear Regression and KNN models. 
Data from a housing price CSV file. 
It includes visualizations to better understand model performance.
"""

import numpy as np
from utils import load_data
from preprocessors import CategoricalEncoder, StandardScaler
from models import LinearRegression, KNN
from evaluators import Evaluator

def main():
    # Step 1: Load the data
    data_path = "data/Housing.csv"
    data = load_data(data_path, verbose=True)

    if data is None:
        print("Data loading failed. Exiting the program.")
        return

    # Step 2: Separate features and target variable
    target_column = 'price'
    x = data.drop(columns=[target_column])
    y = data[target_column].values

    # Step 3: Preprocess the data
    # Encode categorical variables
    encoder = CategoricalEncoder()
    x_encoded = encoder.fit_transform(x)

    # Scale numerical features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_encoded)

    # Step 4: Train and evaluate the Linear Regression model
    print("\n--- Linear Regression Model ---")
    linear_model = LinearRegression(verbose=True)
    linear_model.fit(x_scaled, y)

    # Visualize Linear Regression: True vs. Predicted Values
    linear_model.plot_predictions(x_scaled, y)

    # Evaluate the Linear Regression model
    linear_evaluator = Evaluator(linear_model)
    linear_metrics = linear_evaluator.evaluate(x_scaled, y)
    print("Linear Regression Evaluation Results:", linear_metrics)

    # Step 5: Train and evaluate the KNN model
    print("\n--- K-Nearest Neighbors Model ---")
    knn_model = KNN(k=5, verbose=True)
    knn_model.fit(x_scaled, y)

    # Visualize KNN: Scatter Plot of All Relationships
    knn_model.plot_neighbors(x_scaled, y)

    # Evaluate the KNN model
    knn_evaluator = Evaluator(knn_model)
    knn_metrics = knn_evaluator.evaluate(x_scaled, y)
    print("KNN Evaluation Results:", knn_metrics)

if __name__ == "__main__":
    main()
