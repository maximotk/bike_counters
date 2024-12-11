_target_column_name = "log_bike_count"

# Codidy Data
# Codifies the datetime information into several key date-related features
"""
    Parameters:
    data (pd.DataFrame): The input dataframe with a `date` column, which will be converted to datetime.

    Returns:
    pd.DataFrame: The input dataframe with the following additional columns:
                  - `hour`: The hour part of the datetime.
                  - `weekday`: The day of the week (0=Monday, 6=Sunday).
                  - `daymonth`: A combination of the day and month in `DD_MM` format.
                  - `IsHoliday`: A boolean indicating whether the day is a holiday in France.
    """
# ==============================================================================
def codify_date(data):
    fr_holidays = holidays.France()

    data["datetime"] = pd.to_datetime(data["date"])
    data["date"] = data["datetime"].dt.date
    data["hour"] = data["datetime"].dt.hour
    data["weekday"] = data["datetime"].dt.weekday
    data["daymonth"] = data["datetime"].dt.strftime('%d') + "_" + data["datetime"].dt.month.astype(str)
    data["IsHoliday"] = data["datetime"].dt.date.apply(lambda x: x in fr_holidays)
    
    return data

# Codidy Data
# Extracts: Datetime, Date, Year, Month, Day, Day of Week, Hour, Is Weekend, Is Holiday
"""
    Parameters:
    data (pd.DataFrame): The input dataframe with a column `datetime` containing date and time information.

    Returns:
    pd.DataFrame: The input dataframe with additional date-related features, including:
                  - `year`: The year of the datetime.
                  - `month`: The month of the datetime.
                  - `day`: The day of the datetime.
                  - `day_of_week`: The day of the week (0=Monday, 6=Sunday).
                  - `hour`: The hour of the datetime.
                  - `is_weekend`: A boolean flag indicating whether the day is a weekend.
                  - `IsHoliday`: A boolean flag indicating whether the day is a public holiday in France.
"""
# ==============================================================================
def codify_date_2(data):
    fr_holidays = holidays.France()

    data["datetime"] = data["date"]
    data["date"] = data["datetime"].dt.date
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['day_of_week'] = data['datetime'].dt.dayofweek  # Monday = 0, Sunday = 6
    data['hour'] = data['datetime'].dt.hour
    data['is_weekend'] = data['datetime'].dt.dayofweek >= 5
    data["IsHoliday"] = data["datetime"].dt.date.apply(lambda x: x in fr_holidays)

    return data

# Removes Outliers
# Outliers: For each counter, remove observations for days with zero bike 
#           counts throughout the entire day.
"""
    Parameters:
    data (pd.DataFrame): The input dataframe containing columns `counter_name`, `datetime`, and `log_bike_count`.

    Returns:
    pd.DataFrame: A cleaned version of the input dataframe with observations removed where the bike count is zero for an entire day.
"""
# ==============================================================================
def remove_outliers(data):
    data["date_truncated"] = data["datetime"].dt.floor("D")

    cleaned_data = (
        data.groupby(["counter_name", "date_truncated"])
        ["log_bike_count"].sum()
        .to_frame()
        .reset_index()
        .query("log_bike_count == 0")
        [["counter_name", "date_truncated"]]
        .merge(data, on=["counter_name", "date_truncated"], how="right", indicator=True)
        .query("_merge == 'right_only'")
        .drop(columns=["_merge", "date_truncated"])
    )

    return cleaned_data

# Add Covid Data with 1 during Lockdowns and 0 otherwise
# Quarantine Periods: 2020-10-30", "2020-12-15
#                     2021-03-20", "2021-05-19
"""
    Parameters:
    data (pd.DataFrame): The input dataframe containing a 'datetime' column, which is used to assign the 'Covid-19' label.

    Returns:
    pd.DataFrame: The input dataframe with an added 'Covid-19' column (binary: 1 for COVID-19 periods, 0 otherwise).
    """
# ==============================================================================
def covid_19(data):
    date_ranges = [
        ("2020-10-30", "2020-12-15"),
        ("2021-03-20", "2021-05-19"),
    ]

    data["Covid-19"] = 0 
    for start, end in date_ranges:
        data["Covid-19"] |= data["datetime"].between(start, end).astype(int)

    return data

# Add Covid-19 Stringency Index
'''Enrich a dataset with COVID-19 stringency index data for France.

    Parameters
    ----------
    data : DataFrame
        The input DataFrame containing a 'date' column.

    Returns
    -------
    data : DataFrame
        The input DataFrame merged with France's stringency index data based on the 'date' column.

    Source
    -------
    Thomas Hale, Noam Angrist, Rafael Goldszmidt, Beatriz Kira, Anna Petherick,
    Toby Phillips, Samuel Webster, Emily Cameron-Blake, Laura Hallas, Saptarshi Majumdar,
    and Helen Tatlow. (2021). “A global panel database of pandemic policies (Oxford
    COVID-19 Government Response Tracker).” Nature Human Behaviour.
    https://doi.org/10.1038/s41562-021-01079-8
   
    Stringency Index description:
    https://github.com/OxCGRT/covid-policy-dataset/blob/main/documentation_and_codebook.md#calculation-of-policy-indices
    '''
# ====================================================================================================================
def covid_19_2(data):
    data["date"] = pd.to_datetime(data["date"])
    min = data["date"].min().strftime('%Y-%m-%d')
    max = data["date"].max().strftime('%Y-%m-%d')
    
    covid_19_index = pd.read_csv(Path("data") / "Covid_19_Index.csv")
    covid_19_index["date"] = pd.to_datetime(covid_19_index["Date"], format='%Y%m%d')

    data = (
        covid_19_index
        .query("CountryName == 'France'")
        .query(f"date >= '{min}' and date <= '{max}'")
        [["date", "StringencyIndex_Average"]]
        .merge(data, on=["date"], how="right")
    )

    return data

# Add Weather Data (external data)
# Columns: 't', 'rr1', 'u', 'ht_neige', 'raf10', 'ff', 'ww', 'etat_sol', 'tend'
"""
    Parameters:
    df (pd.DataFrame): The input dataframe containing datetime and other features.

    Returns:
    pd.DataFrame: A dataframe with the original data combined with relevant weather features.
"""
# ==============================================================================
def add_weather(df):
    weather = pd.read_csv('data/external_data.csv')

    weather.drop_duplicates(inplace=True)

    weather = weather[['date', 't', 'rr1', 'u', 'ht_neige', 'raf10', 'ff', 'ww', 'etat_sol', 'tend']]
    
    df['datetime'] = df['datetime'].astype('datetime64[ns]')
    weather['date'] = pd.to_datetime(weather['date'])    
    df['original_index'] = df.index

    df_sorted = df.sort_values('datetime')
    weather_sorted = weather.sort_values('date')
    
    df_merged = pd.merge_asof(df_sorted, weather_sorted, left_on='datetime', right_on='date', direction='backward', suffixes=('', '_weather'))
    
    df_merged = df_merged.sort_values('original_index').set_index('original_index')
    
    
    df_merged = df_merged.drop(columns=['date_weather'])
    
    return df_merged

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


# Get X and y 
'''
    Parameters:
    data (pd.DataFrame): The input dataframe containing the data to process.

    Returns:
    X_df (pd.DataFrame): A dataframe containing the features used for modeling.
    y_array (np.array): A numpy array containing the target values.
'''
# ==============================================================================
def get_X_y(data):
    data = data.drop(columns=["counter_id", "site_id", "site_name", 
                              "bike_count", "counter_installation_date", 
                              "coordinates", "counter_technical_id",
                              "latitude", "longitude", "datetime", "date"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array


# Transforms a Cyclical Feature into Sine and Cosine Components
"""
    Parameters:
    df (pd.DataFrame): The dataframe containing the cyclical feature.
    col (str): The name of the column containing the cyclical feature to transform.
    period (int): The period of the cyclical feature,

    Returns:
    pd.DataFrame: The dataframe with the original column replaced by its sine and cosine transformations.
"""
# ==============================================================================
def cyclic_transform(df, col, period):
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    df = df.drop(columns=col)
    return df

# Applies One-Hot Encoding to specified columns in the dataframe
"""
    Parameters:
    df (pd.DataFrame): The dataframe containing the columns to encode.
    cols (list of str): A list of column names to apply One-Hot Encoding on.

    Returns:
    pd.DataFrame: A dataframe with the original columns replaced by their One-Hot Encoded representations.
    """
# ==============================================================================
def one_hot_encode(df, cols):
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_array = encoder.fit_transform(df[cols])
    encoded_cols = encoder.get_feature_names_out(cols)
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df_encoded = df_encoded.astype(float)
    df = df.drop(columns=cols).join(df_encoded)
    return df