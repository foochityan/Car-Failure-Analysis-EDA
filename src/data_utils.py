import numpy as np
import pandas as pd
import sqlite3 as sq3

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(data):
    # Clean Membership column
    data["Membership"] = data["Membership"].fillna("None")

    # Clean Temperature column
    clean_temperature = data["Temperature"].apply(
        lambda temp: float(temp.split()[0])
    )
    binarize_temperature = clean_temperature.apply(
        lambda temp: 
            "High temperature" if temp > 175 
            else "Low temperature"  # Threshold taken from EDA of Temperature
    )
    data["Temperature"] = binarize_temperature

    # Drop Car ID column
    data = data.drop(columns = "Car ID")

    # Remove remove negative values in RPM
    rpm = data["RPM"]
    remove_idx = data[data["RPM"] < 0].index
    data = data.drop(index = remove_idx)

    return data


def encode(data):
    target_columns = [column for column in data.columns if "Failure" in column]
    feature_columns = [column for column in data.columns if column not in target_columns]

    numerical_columns = data[feature_columns].select_dtypes(exclude = [object]).columns
    categorical_columns = data[feature_columns].select_dtypes(include = [object]).columns

    # Encode categorical columns
    encoded_categoricals = data[categorical_columns].copy()
    encoded_names = []
    for column in categorical_columns:
        if column == "Model":
            # Encode Model column using mean weight encoding
            subset = data["Model"]
            num_rows = data.shape[0]
            
            weights = subset.value_counts(normalize = True)
            encoded = subset.replace(weights)

            encoded_categoricals["Model"] = encoded
            continue

        # Encode all other columns using onehot encoding
        subset = data[column]
        dummy = pd.get_dummies(subset)
        encoded_categoricals = encoded_categoricals.drop(columns = column)

        encoded_categoricals = pd.concat([encoded_categoricals, dummy], axis = 1)

    # Combine failure columns
    fault_name_to_label = {
        "Failure A": 1,
        "Failure B": 2,
        "Failure C": 3,
        "Failure D": 4,
        "Failure E": 5
    }

    targets = data.loc[:, target_columns]
    for column in target_columns:
        target = targets[column]
        failure_name = column.split()[1]
        label_name = ord(failure_name) - 64

        encoded = target.replace({1: label_name})
        targets[column] = encoded

    new_target = targets.sum(axis = 1)
    new_target.name = "Faults"

    # Join all numerical, categorical and target columns back
    encoded_data = pd.concat([
        data[numerical_columns],
        encoded_categoricals,
        new_target
    ], axis = 1)

    return encoded_data.reset_index(drop = True)


def prepare(data, config):
    if config.transform:
        # If user wants to transform fuel consumption
        data["Fuel consumption"] = np.log(data["Fuel consumption"])

    if config.balance:
        # If user wants to balance target labels
        target = data["Faults"]

        # Calculate the proportion of each label
        props = target.value_counts(normalize = True)

        # Calculate the proportion of cars with faults
        fault_props = props[props.index != 0]
        average_prop = np.mean(fault_props)
        
        # Split the data into cars with and without faults
        fault_data_idx = target[target > 0].index
        nofault_data_idx = target[target == 0].index
        
        # Calculate the sample amount from cars without faults
        nofault_sample_perc = average_prop / props.loc[0]
        nofault_sample_num = int(nofault_sample_perc * len(nofault_data_idx))

        # Randomly sample indices to subset the whole data
        sampled_nofaults_idx = np.random.choice(nofault_data_idx, size = nofault_sample_num)

        # Combined the desired indices and subset the data
        sampled_data_idx = np.concatenate([sampled_nofaults_idx, fault_data_idx], axis = 0)
        data = data.iloc[sampled_data_idx, :]


    # Split the data into features and target
    target_column = "Faults"
    features = data.drop(columns = target_column)
    target = data.loc[:, target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        target, 
        test_size = config.test_size, 
        stratify = target,      # Ensures equal weighted labels in train and test
        random_state = config.seed
    )

    if config.scale:
        # If user wants to scale the data
        scaler = StandardScaler()
        scaler.fit(X_train)                     # Fit only on train
        X_train = scaler.transform(X_train)     # Transform both train and test sits
        X_test = scaler.transform(X_test)
    
    y_train = y_train.ravel().astype(float)
    y_test = y_test.ravel().astype(float)

    return X_train, X_test, y_train, y_test


def read_data(data_path):
    data_name = data_path.rstrip(".db")

    # Connecting to database
    connection = sq3.connect(data_path)
    cursor = connection.cursor()
    query = """
        SELECT *
        FROM 
    """

    # Fetching the data from the database
    cursor.execute(query + data_name)
    raw_data = cursor.fetchall()
    columns = [
        "Car ID", "Model", "Color", "Temperature",
        "RPM", "Factory", "Usage", "Fuel consumption",
        "Membership", "Failure A", "Failure B", "Failure C",
        "Failure D", "Failure E"
    ]

    data = pd.DataFrame(raw_data, columns = columns)
    data = encode(clean_data(data))
    return data
    