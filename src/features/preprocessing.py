import numpy as np
import pandas as pd
from typing import Generator, Tuple
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import TimeSeriesSplit

_target_column_name = "log_bike_count"


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outlier days where the bike count is zero for the entire day.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing at least:
        - 'counter_name'
        - 'datetime'
        - 'log_bike_count'

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with full-zero days removed.
    """
    data["date_truncated"] = data["datetime"].dt.floor("D")

    cleaned_data = (
        data.groupby(["counter_name", "date_truncated"])["log_bike_count"]
        .sum()
        .to_frame()
        .reset_index()
        .query("log_bike_count == 0")
        [["counter_name", "date_truncated"]]
        .merge(data, on=["counter_name", "date_truncated"], how="right", indicator=True)
        .query("_merge == 'right_only'")
        .drop(columns=["_merge", "date_truncated"])
    )

    return cleaned_data

def handle_missing_values(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    """
    Handle missing values in the dataframe by applying interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least:
        - 'datetime' (used for sorting before interpolation)
        - other numeric columns with potential missing values
    method : str, optional
        Interpolation method to use (default is "linear").
        Other options can be provided as supported by pandas.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values interpolated.
    """
    df = df.sort_values(by="datetime").reset_index(drop=True)

    missing_info = df.isnull().sum()
    missing_columns = missing_info[missing_info > 0]

    if missing_columns.empty:
        return df

    interpolated_df = df.copy()
    for col in missing_columns.index:
        interpolated_df[col] = interpolated_df[col].interpolate(method=method, axis=0)

    return interpolated_df

def get_X_y(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Split dataframe into features (X) and target (y).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing the target column 'log_bike_count'.

    Returns
    -------
    X_df : pd.DataFrame
        Feature dataframe with non-predictive columns removed.
    y_array : np.ndarray
        Target array (log_bike_count).
    """
    data = data.drop(
        columns=[
            "counter_id", "site_id", "site_name",
            "bike_count", "counter_installation_date",
            "coordinates", "counter_technical_id",
            "latitude", "longitude", "datetime", "date",
        ]
    )
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array

def get_cv(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate train-test indices for time series cross-validation with random subsampling of the test set.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    random_state : int, default=0
        Seed for reproducible random sampling of test indices.

    Yields
    ------
    train_idx : np.ndarray
        Indices for the training set.
    test_idx : np.ndarray
        Randomly sampled indices for the test set (approximately one-third of the original test set),
        ensuring that test samples are not consecutive.

    Returns
    -------
    Generator[Tuple[np.ndarray, np.ndarray], None, None]
        Generator yielding tuples of training and test indices for each fold.
    """
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X, y):
        # Take a random sampling on test_idx so that samples are not consecutive
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def cyclic_transform(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """
    Encode a cyclical feature into sine and cosine components.

    Parameters
    ---------- 
    df : pd.DataFrame
        Input dataframe containing the cyclical column.
    col : str
        Column name to encode.
    period : int
        Period of the cycle (e.g. 24 for hours in a day).

    Returns
    -------
    pd.DataFrame
        Dataframe with sine and cosine features replacing the original column.
    """
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df.drop(columns=col)


def one_hot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing categorical columns.
    cols : list of str
        List of categorical columns to encode.

    Returns
    -------
    pd.DataFrame
        Dataframe with categorical variables replaced by their encoded representation.
    """
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_array = encoder.fit_transform(df[cols])
    encoded_cols = encoder.get_feature_names_out(cols)

    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index).astype(float)
    return df.drop(columns=cols).join(df_encoded)


def sin_transformer(period: int) -> FunctionTransformer:
    """
    Create a sklearn transformer to apply sine encoding for cyclical features.
    """
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period: int) -> FunctionTransformer:
    """
    Create a sklearn transformer to apply cosine encoding for cyclical features.
    """
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
