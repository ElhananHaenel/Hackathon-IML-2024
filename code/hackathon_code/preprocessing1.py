import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

def replace_infrequent_categories(data, categorical_features, threshold=0.01):
    """
    Replace infrequent categories with 'Other'.

    Parameters:
    data (DataFrame): The input data.
    categorical_features (list): List of categorical feature names to process.
    threshold (float): The minimum frequency threshold for a category to be kept.

    Returns:
    DataFrame: The data with infrequent categories replaced by 'Other'.
    """
    for col in categorical_features:
        # Calculate the frequency of each category
        freq = data[col].value_counts(normalize=True)
        # Determine which categories are infrequent
        infrequent_categories = freq[freq < threshold].index
        # Replace infrequent categories with 'Other'
        data[col] = data[col].apply(lambda x: 'Other' if x in infrequent_categories else x)
    return data

def one_hot_encode(data, categorical_features):
    """
    One-hot encode the categorical features in the dataset.

    Parameters:
    data (DataFrame): The input data.
    categorical_features (list): List of categorical feature names to encode.

    Returns:
    DataFrame: The one-hot encoded data.
    """
    data = replace_infrequent_categories(data, categorical_features)
    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    return data_encoded

def turn_into_categorical(data):
    """
    Convert specified features into categorical features and one-hot encode them.

    Parameters:
    data (DataFrame): The input data.

    Returns:
    DataFrame: The data with categorical features one-hot encoded.
    """
    features = [
        'trip_id', 'part', 'trip_id_unique_station', 'trip_id_unique',
        'line_id', 'direction', 'alternative', 'cluster',
        'station_index', 'station_id', 'station_name', 'arrival_time',
        'door_closing_time', 'arrival_is_estimated', 'latitude', 'longitude',
        'passengers_up', 'passengers_continue', 'mekadem_nipuach_luz',
        'passengers_continue_menupach'
    ]
    target = 'passengers_up'

    categorical_features = ['trip_id', 'part', 'station_name', 'trip_id_unique_station',
                            'trip_id_unique', 'line_id','arrival_time',
                            'door_closing_time', 'direction',
                            'alternative', 'cluster', 'station_id']

    # One-hot encode categorical features
    data_encoded = one_hot_encode(data, categorical_features)
    return data_encoded

def remove_irrelevant_columns(data):
    """
    Remove columns that are not relevant for the analysis.

    Parameters:
    data (DataFrame): The input data.

    Returns:
    DataFrame: The data with irrelevant columns removed.
    """
    #data = data.drop(columns=['trip_id_unique', 'station_name'])
    return data

def invalid_features_test(data):
    """
    Replace invalid feature values with predefined replacement values for the test data.

    Parameters:
    data (DataFrame): The input data.

    Returns:
    DataFrame: The data with invalid feature values replaced.
    """
    replacement_values = {
        'trip_id': np.nan,
        'line_id': np.nan,
        'direction': np.nan,
        'station_index': np.nan,
        'station_id': np.nan,
        'arrival_time': pd.NaT,
        'door_closing_time': pd.NaT,
        'latitude': np.nan,
        'longitude': np.nan,
        'passengers_continue': 0,
        'mekadem_nipuach_luz': np.nan,
        'passengers_continue_menupach': np.nan
    }

    # Validate and replace invalid values
    data['trip_id'] = data['trip_id'].apply(lambda x: x if str(x).isdigit() else replacement_values['trip_id'])
    data['line_id'] = data['line_id'].apply(lambda x: x if str(x).isdigit() else replacement_values['line_id'])
    data['direction'] = data['direction'].apply(lambda x: x if x in [1, 2] else replacement_values['direction'])
    data['station_index'] = data['station_index'].apply(lambda x: x if str(x).isdigit() else replacement_values['station_index'])
    data['station_id'] = data['station_id'].apply(lambda x: x if str(x).isdigit() else replacement_values['station_id'])
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M', errors='coerce').fillna(replacement_values['arrival_time'])
    data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M', errors='coerce').fillna(replacement_values['door_closing_time'])
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce').fillna(replacement_values['latitude'])
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce').fillna(replacement_values['longitude'])
    data['passengers_continue'] = data['passengers_continue'].fillna(0).apply(lambda x: x if x >= 0 else replacement_values['passengers_continue'])
    data['mekadem_nipuach_luz'] = pd.to_numeric(data['mekadem_nipuach_luz'], errors='coerce').fillna(replacement_values['mekadem_nipuach_luz'])
    data['passengers_continue_menupach'] = pd.to_numeric(data['passengers_continue_menupach'], errors='coerce').fillna(replacement_values['passengers_continue_menupach'])

    return data

def invalid_features_train(data):
    """
    Remove rows with invalid feature values for the training data.

    Parameters:
    data (DataFrame): The input data.

    Returns:
    DataFrame: The data with invalid rows removed.
    """
    data = data[data['trip_id'].astype(str).str.isdigit()]
    data = data[data['line_id'].astype(str).str.isdigit()]
    data = data[data['direction'].isin([1, 2])]
    data = data[data['station_index'].astype(str).str.isdigit()]
    data = data[data['station_id'].astype(str).str.isdigit()]
    data = data[pd.to_numeric(data['latitude'], errors='coerce').notnull()]
    data = data[pd.to_numeric(data['longitude'], errors='coerce').notnull()]
    data = data[data['passengers_up'].fillna(0).astype(int) >= 0]
    data = data[data['passengers_continue'].fillna(0).astype(int) >= 0]
    data = data[pd.to_numeric(data['mekadem_nipuach_luz'], errors='coerce').notnull()]
    data = data[pd.to_numeric(data['passengers_continue_menupach'], errors='coerce').notnull()]

    return data

def add_relevant_columns(data):
    """
    Add new columns with calculated values relevant to the analysis.

    Parameters:
    data (DataFrame): The input data.

    Returns:
    DataFrame: The data with new relevant columns added.
    """
    max_station_index = data.groupby('line_id')['station_index'].max().reset_index()
    max_station_index.columns = ['line_id', 'max_station_index']

    # Merge the result back into the original DataFrame
    data = pd.merge(data, max_station_index, on='line_id', how='left')
    data['station_index_ratio'] = data['station_index'] / data['max_station_index']
    data.drop(columns=['max_station_index'], inplace=True)

    # Calculate number of lines per station_id
    num_lines_in_stat = data.groupby('station_id')['line_id'].nunique().reset_index()
    num_lines_in_stat.columns = ['station_id', 'num_lines_in_stat']

    # Merge the result back into the original DataFrame
    data = pd.merge(data, num_lines_in_stat, on='station_id', how='left')
    
    return data

def preprocess_train(data):
    """
    Preprocess the training data by removing duplicates, invalid features, and irrelevant columns.

    Parameters:
    data (DataFrame): The input training data.

    Returns:
    tuple: Tuple containing the preprocessed features (X) and target variable (y).
    """
    data.drop_duplicates(inplace=True)
    data = invalid_features_train(data)
    data = turn_into_categorical(data)
    data = remove_irrelevant_columns(data)

    target = 'passengers_up'
    if 'passengers_up' in data.columns:
        y = data[target]
        data = data.drop(columns=['passengers_up'])
    else:
        y = pd.DataFrame()

    return data, y

def preprocess_test(data):
    """
    Preprocess the test data by replacing invalid feature values, and removing irrelevant columns.

    Parameters:
    data (DataFrame): The input test data.

    Returns:
    tuple: Tuple containing the preprocessed features (X) and target variable (y).
    """
    data = invalid_features_test(data)
    data = turn_into_categorical(data)
    data = remove_irrelevant_columns(data)

    target = 'passengers_up'
    if 'passengers_up' in data.columns:
        y = data[target]
        data = data.drop(columns=['passengers_up'])
    else:
        y = pd.DataFrame()

    return data, y
