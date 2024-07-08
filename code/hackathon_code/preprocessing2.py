


# Task 2:  
# - Add column: trip_duration, number of stations, passengers_up_per_station, total_passenger_up, 
# passenger_down_per_station, total_passenger_down, distance_between_stations, total_distance,

# - Delete: arrival_time for stations, trip_id_unique_station, station_id, station_name, station_index,
# door_closing_time, latitude, longitude

# - Maybe delete: alternative, arrival_is_estimated, 

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from category_encoders import OrdinalEncoder, TargetEncoder
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
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
    features = [
        'trip_id', 'part', 'trip_id_unique_station', 'trip_id_unique',
    'line_id', 'direction', 'alternative', 'cluster',
    'station_index', 'station_id', 'station_name', 'arrival_time',
    'door_closing_time', 'arrival_is_estimated', 'latitude', 'longitude',
    'passengers_up', 'passengers_continue', 'mekadem_nipuach_luz',
    'passengers_continue_menupach'
    ]
    target = 'passengers_up'

    categorical_features = ['trip_id', 'part', 'station_name', 'trip_id_unique_station', 'line_id', 'direction',
                            'alternative', 'cluster', 'station_id']

    # Encode categorical features using target encoding
    # encoder = TargetEncoder(cols=categorical_features)
    # data_encoded = encoder.fit_transform(data[categorical_features],
    #                                      data[target])
    train_data_encoded = one_hot_encode(data, categorical_features)
    return train_data_encoded

def invalid_features_test(data):
    # Define replacement values for invalid data
    replacement_values = {
        'trip_id': np.nan,
        'line_id': np.nan,
        'direction': np.nan,
        'station_index': np.nan,
        'station_id': np.nan,
        'arrival_time': pd.NaT,
        'door_closing_time': pd.NaT,
        #'arrival_is_estimated': np.nan,
        'latitude': np.nan,
        'longitude': np.nan,
        'passengers_up': 0,
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
    #data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce').fillna(replacement_values['arrival_time'])
    #data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S', errors='coerce').fillna(replacement_values['door_closing_time'])
    #data['arrival_is_estimated'] = data['arrival_is_estimated'].map({'TRUE': 1, 'FALSE': 0}).fillna(replacement_values['arrival_is_estimated'])
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce').fillna(replacement_values['latitude'])
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce').fillna(replacement_values['longitude'])
    data['passengers_up'] = data['passengers_up'].fillna(0).apply(lambda x: x if x >= 0 else replacement_values['passengers_up'])
    data['passengers_continue'] = data['passengers_continue'].fillna(0).apply(lambda x: x if x >= 0 else replacement_values['passengers_continue'])
    data['mekadem_nipuach_luz'] = pd.to_numeric(data['mekadem_nipuach_luz'], errors='coerce').fillna(replacement_values['mekadem_nipuach_luz'])
    data['passengers_continue_menupach'] = pd.to_numeric(data['passengers_continue_menupach'], errors='coerce').fillna(replacement_values['passengers_continue_menupach'])

    return data


def invalid_features_train(data):
    # Check if trip_id is numeric
    data = data[data['trip_id'].astype(str).str.isdigit()]

    # Check if line_id is numeric
    data = data[data['line_id'].astype(str).str.isdigit()]

    # Check if direction is 1 or 2
    data = data[data['direction'].isin([1, 2])]

    # Check if station_index is numeric
    data = data[data['station_index'].astype(str).str.isdigit()]

    # Check if station_id is numeric
    data = data[data['station_id'].astype(str).str.isdigit()]

    # Check if latitude is numeric
    data = data[pd.to_numeric(data['latitude'], errors='coerce').notnull()]

    # Check if longitude is numeric
    data = data[pd.to_numeric(data['longitude'], errors='coerce').notnull()]

    # Check if passengers_up is non-negative integer
    data = data[data['passengers_up'].fillna(0).astype(int) >= 0]

    # Check if passengers_continue is non-negative integer
    data = data[data['passengers_continue'].fillna(0).astype(int) >= 0]

    # Check if mekadem_nipuach_luz is numeric
    data = data[
        pd.to_numeric(data['mekadem_nipuach_luz'], errors='coerce').notnull()]

    # Check if passengers_continue_menupach is numeric
    data = data[pd.to_numeric(data['passengers_continue_menupach'],
                              errors='coerce').notnull()]

    return data






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

   

def invalid_features(data: pd.DataFrame):
    categorical_features = [
                            'cluster', 'station_id', 'line_id']
    numerical_features = ['latitude', 'longitude', 'passengers_up', 'passengers_continue' ]
    data[categorical_features].fillna(data[categorical_features].mode().iloc[0], inplace=True)
    data[numerical_features].fillna(data[numerical_features].mean(), inplace=True)
    train_data_encoded = one_hot_encode(data, categorical_features)
    return train_data_encoded


def process_trip_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:    
    # Calculate the new features
    df['distance_from_previous'] = 0
    grouped = df.groupby('trip_id_unique')

    # Define a function to calculate distances within each trip
    def calculate_distances(trip):
        trip = trip.sort_values(by='station_index').reset_index(drop=True)
        distances = [0]  # First station in each trip has 0 distance from previous
        for i in range(1, len(trip)):
            prev_station = (trip.loc[i-1, 'latitude'], trip.loc[i-1, 'longitude'])
            current_station = (trip.loc[i, 'latitude'], trip.loc[i, 'longitude'])
            # print(trip)
            distance = geodesic(prev_station, current_station).kilometers
            distances.append(distance)
        trip['distance_from_previous'] = distances
        return trip

# Apply the function to each trip group
    df = grouped.apply(calculate_distances).reset_index(drop=True)
    # Helper functions
    def calculate_trip_duration(arrival_times):
        arrival_times_dt = pd.to_datetime(arrival_times, format='%H:%M:%S')

        # Calculate the difference between max and min arrival times in seconds
        trip_duration_seconds = (
                    arrival_times_dt.max() - arrival_times_dt.min()).total_seconds()

        # Convert trip duration to minutes
        trip_duration_minutes = trip_duration_seconds / 60.0
        return trip_duration_minutes
    
    def calculate_average_distance(distances):
        distances = distances[1:]  # Ignore the first station (distance=0)
        return distances.mean() if len(distances) > 0 else 0
        
    new_df = df.groupby("trip_id_unique").agg(
        number_of_stations=('station_index', 'max'),
        total_passenger_up=('passengers_up', 'sum'),
        distance_between_stations=('distance_from_previous', calculate_average_distance),
        # exit_time=('arrival_time', 'min'),
        trip_duration_in_minutes=('arrival_time', calculate_trip_duration)
    ).reset_index()

    new_df['passengers_up_per_station'] = new_df['total_passenger_up'] / new_df['number_of_stations']
    # new_df['passenger_down_per_station'] = new_df['total_passenger_down'] / new_df['number_of_stations']
    new_df["total_distance"] = new_df["distance_between_stations"] * new_df["number_of_stations"]

    # Adding other required columns from the first occurrence in each group
    first_occurrence = df.groupby('trip_id_unique').first().reset_index()
    deleted_cols = ['trip_id_unique', 'trip_id', 'arrival_time', 'latitude', 'longitude', 'distance_from_previous', 'station_index', 'passengers_up', 'passengers_continue', 
                    'arrival_is_estimated', 'door_closing_time', 'station_name', 'station_id', 'alternative',  'trip_id_unique_station', 'part', 'cluster']
    for col in df.columns:
        if(col in deleted_cols): continue
        new_df[col] = new_df['trip_id_unique'].map(first_occurrence.set_index('trip_id_unique')[col])
    print(new_df.columns)
    
    return new_df


def preprocess_train(X: pd.DataFrame) -> pd.DataFrame:
    # X = remove_irrelevant_columns(X)
    X= invalid_features_train(X)
    X = turn_into_categorical(X)
    X=process_trip_data(X, True)
    target = 'trip_duration_in_minutes'
    if 'trip_duration_in_minutes in X.columns':
        y = X[target]
        X = X.drop(columns=['trip_duration_in_minutes'])
    else:
        y = pd.DataFrame()
    return X, y
    # return X

def preprocess_test(X: pd.DataFrame) -> pd.DataFrame:
    # X = remove_irrelevant_columns(X)

    X= invalid_features_test(X)
    X = turn_into_categorical(X)
    X=process_trip_data(X, False)
    target = 'trip_duration_in_minutes'
    if 'trip_duration_in_minutes in X.columns':
        y = X[target]
        X = X.drop(columns=['trip_duration_in_minutes'])
    else:
        y = pd.DataFrame()
    return X, y
    # return X

