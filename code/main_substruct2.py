import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from argparse import ArgumentParser
import logging
from hackathon_code import preprocessing2
import seaborn as sns
import matplotlib.pyplot as plt

"""
Usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

Example:
    python code/main.py --training_set /cs/usr/gililior/train_data.csv --test_set /cs/usr/gililior/test_data.csv --out predictions/trip_duration_predictions.csv 
"""

# Uncomment the following function if you want to plot the correlation heatmap
# def plot_correlation(data):
#     """
#     Plot a heatmap of correlations between numerical features in the dataset.
#
#     Parameters:
#     data (DataFrame): The input data containing numerical features.
#
#     Returns:
#     None
#     """
#     # Compute the correlation matrix
#     corr_matrix = data.corr()
#
#     # Generate a mask for the upper triangle
#     mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#
#     # Set up the matplotlib figure
#     plt.figure(figsize=(12, 8))
#
#     # Draw the heatmap with the mask
#     sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
#
#     # Customize the plot
#     plt.title('Correlation Matrix of Features')
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#
#     # Show plot
#     plt.show()

def load_data(file_path, encoding='ISO-8859-8'):
    """
    Load data from a file path.

    Parameters:
    file_path (str): The file path to the data file.
    encoding (str): The encoding format of the data file.

    Returns:
    DataFrame: A DataFrame containing the loaded data.
    """
    data = pd.read_csv(file_path, encoding=encoding)
    return data

def preprocess_data(data):
    """
    Preprocess the data by separating features and target variable.

    Parameters:
    data (DataFrame): The input data containing features and target variable.

    Returns:
    tuple: Tuple containing the features (X) and the target variable (y).
    """
    target = 'trip_duration_in_minutes'

    X = data.drop('trip_duration_in_minutes', axis=1)
    y = data[target]

    return X, y

def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model.

    Parameters:
    X_train (DataFrame): Training feature data.
    y_train (Series): Training target data.

    Returns:
    RandomForestRegressor: The trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """
    Predict using the trained model.

    Parameters:
    model (RandomForestRegressor): The trained model.
    X_test (DataFrame): Test feature data.

    Returns:
    ndarray: The predicted values.
    """
    return model.predict(X_test)

def save_predictions(X_test, predictions, output_path, encoding='ISO-8859-8'):
    """
    Save predictions to a CSV file.

    Parameters:
    X_test (DataFrame): Test feature data.
    predictions (ndarray): The predicted values.
    output_path (str): The file path to save the predictions.
    encoding (str): The encoding format of the CSV file.
    """
    output_df = X_test.copy()
    output_df['trip_duration_in_minutes'] = predictions
    output_df.to_csv(output_path, index=False, encoding=encoding)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="Path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="Path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="Path of the output file as required in the task description")
    args = parser.parse_args()

    # Load the training and test data
    logging.info("Loading data and splitting into training and test sets...")
    train_data = load_data(args.training_set)
    test_data = load_data(args.test_set)

    # Extract trip IDs for the test set
    trip_id = pd.DataFrame(test_data['trip_id_unique_station'])

    # Preprocess the training set
    X_train, y_train = preprocessing2.preprocess_train(train_data)

    # Train the model (using RandomForestRegressor)
    logging.info("Training the model...")
    model = train_model(X_train.drop("trip_id_unique", axis=1), y_train)

    # Preprocess the test set
    logging.info("Preprocessing the test set...")
    X_test, y_test = preprocessing2.preprocess_test(test_data)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Predict the test set
    logging.info("Predicting the test set...")
    predictions = predict(model, X_test.drop("trip_id_unique", axis=1))

    # Handle zero predictions by replacing them with the average of non-zero predictions
    non_zero_average = predictions[predictions != 0].mean()
    predictions[predictions == 0] = non_zero_average

    # Save the predictions along with the test set features
    logging.info(f"Saving predictions to {args.out}...")
    write_predictions = pd.DataFrame(predictions, columns=['trip_duration_in_minutes'])
    write_predictions = pd.concat([trip_id.reset_index(drop=True), write_predictions], axis=1)
    write_predictions.to_csv(args.out, index=False)

    logging.info("Predictions saved successfully.")
    
    # Uncomment the following lines if you want to plot the correlation heatmap
    # numerical_data = X_train.select_dtypes(include=[np.number])
    # plot_correlation(numerical_data)
