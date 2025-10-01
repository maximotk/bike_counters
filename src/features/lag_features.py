# Add Lags or Rolling Mean or Rollind Standard Deviation for one group
"""
    Parameters:
    group (pd.DataFrame): A subset of the data, typically grouped by 'counter_id', containing columns like 'datetime' and 'bike_count'.

    Returns:
    pd.DataFrame: The input group with additional lag and rolling features.
"""
# ==============================================================================
def _add_lag_and_rolling_features_group(group):
    group = group.sort_values(by='datetime')
    
    group['lag_1'] = group['bike_count'].shift(1)
    group['lag_24'] = group['bike_count'].shift(24)
    group['lag_168'] = group['bike_count'].shift(168)
    
    group['rolling_mean_24h'] = group['bike_count'].rolling(window=24, min_periods=1).mean()
    group['rolling_std_24h'] = group['bike_count'].rolling(window=24, min_periods=1).std()
    group['rolling_mean_7d'] = group['bike_count'].rolling(window=168, min_periods=1).mean()
    group['rolling_std_7d'] = group['bike_count'].rolling(window=168, min_periods=1).std()

    return group

# Add Lags or Rolling Mean or Rollind Standard Deviation
"""
    Parameters:
    data (pd.DataFrame): The input dataframe containing the data with a 'counter_id' and 'datetime' column.

    Returns:
    pd.DataFrame: The transformed dataframe with added lag and rolling features, filtered by datetime.
"""
# ==============================================================================
def add_lag_and_rolling_features(data):
    data_lag_rolling = (
        data.groupby('counter_id')
        .apply(_add_lag_and_rolling_features_group)
        .reset_index(drop=True)
        .query(f"datetime > '{data['datetime'].min() + pd.offsets.Week()}'")
    )
    return data_lag_rolling