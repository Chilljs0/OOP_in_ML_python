"""
data_loader.py

Functions for loading data (CSV) in an ETL process.
"""

import pandas as pd

def load_data(filepath, verbose=False):
    """
    Loads a dataset from a CSV file and returns a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.
        verbose (bool): True prints detailed information about the loaded data.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        if verbose:
            print(f"Data loaded successfully from {filepath}")
            print("First 5 rows of the data:")
            print(data.head())
            print("\nData Summary:")
            print(data.info())
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None
