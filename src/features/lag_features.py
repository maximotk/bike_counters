import pandas as pd

def _add_lag_and_rolling_features_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling statistical features to a grouped time series.

    Parameters
    ----------
    group : pd.DataFrame
        A subset of the data (typically grouped by 'counter_id'),
        containing 'datetime' and 'bike_count' columns.

    Returns
    -------
    pd.DataFrame
        The same group with additional lag and rolling features:
        - lag_1, lag_24, lag_168
        - rolling_mean_24h, rolling_std_24h
        - rolling_mean_7d, rolling_std_7d
    """
    group = group.sort_values(by="datetime")

    group["lag_1"] = group["bike_count"].shift(1)
    group["lag_24"] = group["bike_count"].shift(24)
    group["lag_168"] = group["bike_count"].shift(168)

    group["rolling_mean_24h"] = group["bike_count"].rolling(window=24, min_periods=1).mean()
    group["rolling_std_24h"] = group["bike_count"].rolling(window=24, min_periods=1).std()
    group["rolling_mean_7d"] = group["bike_count"].rolling(window=168, min_periods=1).mean()
    group["rolling_std_7d"] = group["bike_count"].rolling(window=168, min_periods=1).std()

    return group


def add_lag_and_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lag and rolling statistical feature generation across all groups.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing at least:
        - 'counter_id'
        - 'datetime'
        - 'bike_count'

    Returns
    -------
    pd.DataFrame
        Transformed dataframe with lag and rolling features added,
        filtered to exclude the first week (to avoid incomplete lags).
    """
    data_lag_rolling = (
        data.groupby("counter_id")
        .apply(_add_lag_and_rolling_features_group)
        .reset_index(drop=True)
        .query(f"datetime > '{data['datetime'].min() + pd.offsets.Week()}'")
    )
    return data_lag_rolling
