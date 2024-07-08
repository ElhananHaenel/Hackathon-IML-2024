import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser
import logging
from category_encoders import TargetEncoder
import warnings
from sklearn.model_selection import train_test_split
from hackathon_code import preprocessing1

"""
Usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

Example:
    python code/main.py --training_set /cs/usr/gililior/train_data.csv --test_set /cs/usr/gililior/test_data.csv --out predictions/trip_duration_predictions.csv 
"""

def calculate_mse(actual, predicted):
    """
    Calculate the Mean Squared Error (MSE) between actual and predicted values.

    Parameters:
    actual (array-like): Actual values of the target variable.
    predicted (array-like): Predicted values of the target variable.

    Returns:
    float: The Mean Squared Error (MSE).
    """
    mse = mean_squared_error(actual, predicted)
    return mse

def load_data(file_path, encoding='ISO-8859-8'):
    """
    Load data from a file path.

    Parameters:
    file_path (str): The file path to the data file.
    encoding (str): The encoding format of the data file.

    Returns:
    DataFrame: A DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path, encoding=encoding)

def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model.

    Parameters:
    X_train (DataFrame): Training feature data.
    y_train (Series): Training target data.

    Returns:
    RandomForestRegressor: The trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use RandomForestRegressor
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """
    Predict using the trained model and round predictions.

    Parameters:
    model (RandomForestRegressor): The trained model.
    X_test (DataFrame): Test feature data.

    Returns:
    ndarray: The rounded predictions.
    """
    predictions = model.predict(X_test)
    rounded_predictions = np.round(predictions)
    rounded_predictions[rounded_predictions < 0] = 0
    return rounded_predictions

def save_predictions(trip_ids, predictions, output_path, encoding='ISO-8859-8'):
    """
    Save predictions to a CSV file.

    Parameters:
    trip_ids (Series): Trip IDs associated with the predictions.
    predictions (ndarray): The predicted values.
    output_path (str): The file path to save the predictions.
    encoding (str): The encoding format of the CSV file.
    """
    output_df = pd.DataFrame({
        'trip_id_unique_station': trip_ids,
        'predictions': predictions
    })
    output_df.to_csv(output_path, index=False, encoding=encoding)

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="Path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="Path to the test set")
    parser.add_argument('--out', type=str, required=True, help="Path of the output file as required in the task description")
    args = parser.parse_args()

    # Load the training and test data
    logging.info("Loading training data...")
    train_data = load_data(args.training_set)

    logging.info("Loading test data...")
    test_data = load_data(args.test_set)

    # Extract trip IDs for the test set
    trip_id = test_data['trip_id_unique_station']

    # Preprocess the training set
    X_train, y_train = preprocessing1.preprocess_train(train_data)

    # Train the model (using RandomForestRegressor)
    logging.info("Training the model...")
    model = train_model(X_train, y_train)

    # Preprocess the test set
    logging.info("Preprocessing the test set...")
    X_test, y_test = preprocessing1.preprocess_test(test_data)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Predict the test set
    logging.info("Predicting the test set...")
    predictions = predict(model, X_test)

    # Save the predictions along with the test set features
    logging.info(f"Saving predictions to {args.out}...")
    save_predictions(trip_id, predictions, args.out)

    logging.info("Predictions saved successfully.")
    warnings.simplefilter(action='default', category=FutureWarning)
