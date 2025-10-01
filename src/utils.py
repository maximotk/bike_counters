problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X, y):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)

def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

# Handles missing values in a dataframe by applying an interpolation method
"""
    Parameters:
    df (pd.DataFrame): The input dataframe with potential missing values.
    method (str): The interpolation method to use (default is "linear"). This can be adjusted as needed.

    Returns:
    pd.DataFrame: The dataframe with missing values handled.
    """
# ==============================================================================
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
