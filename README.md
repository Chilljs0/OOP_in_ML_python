# OOP_in_ML_python
**IN PROGRESS** Project in Python that contains a simple, custom OOP based implementations of Machine Learning models.


## Project Structure

Breakdown of each folder and file in the project.

### Folders and Files

- `data/`: Folder for storing datasets.
  - `Housing.csv`: Example dataset.

- `models/`: Contains implementations of machine learning models.
  - `__init__.py`: Initializes the models package.
  - `linear_regression.py`: Implementation of a custom Linear Regression model with verbose mode and documentation.
  - `knn.py`: Implementation of a custom K-Nearest Neighbors (KNN) model with verbose mode and documentation.

- `preprocessors/`: Contains classes for preprocessing steps.
  - `__init__.py`: Initializes the preprocessors package.
  - `encoder.py`: A class for handling categorical encoding.
  - `scaler.py`: A class for feature scaling.

- `evaluators/`: Contains evaluation metrics and evaluation logic.
  - `__init__.py`: Initializes the evaluators package.
  - `metrics.py`: Contains functions for calculating evaluation metrics.
  - `evaluator.py`: A class for evaluating model performance using custom metrics.

- `utils/`: Utility functions for data loading and other common tasks.
  - `__init__.py`: Initializes the utils package.
  - `data_loader.py`: Contains a function to load data from a CSV file.

- `main.py`: The main script to run the pipeline. It loads the data, preprocesses it, trains the models, and evaluates their performance.

- `requirements.txt`: Lists the required Python packages for the project.
