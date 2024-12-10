import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X, y):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)
    #splits = []
    #for train_idx, test_idx in cv.split(data):
        # Take a random sampling on test_idx so that samples are not consecutive
    #    splits.append((train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)))
    
    # Return the last 3 splits
    #return splits[-3:]

def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

def handle_missing_values(df, method):
    df = df.sort_values(by="datetime").reset_index(drop=True)

    missing_info = df.isnull().sum()
    missing_columns = missing_info[missing_info > 0]
    
    if missing_columns.empty:
        print("No missing values detected.")
        return df
    
    # Print columns with missing values and their counts
    print("Columns with missing values and their counts:")
    print(missing_columns)
    
    # Replace missing values only in columns with missing data
    interpolated_df = df.copy()
    for col in missing_columns.index:
        interpolated_df[col] = interpolated_df[col].interpolate(method='linear', axis=0)
    
    return interpolated_df
