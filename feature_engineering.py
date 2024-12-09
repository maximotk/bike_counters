import pandas as pd
import numpy as np
import holidays
from pathlib import Path

_target_column_name = "log_bike_count"

# Why only this? Why not month and day as well?
def codify_date(data):
    fr_holidays = holidays.France()

    data["datetime"] = pd.to_datetime(data["date"])
    data["date"] = data["datetime"].dt.date
    data["hour"] = data["datetime"].dt.hour
    data["weekday"] = data["datetime"].dt.weekday
    data["daymonth"] = data["datetime"].dt.strftime('%d') + "_" + data["datetime"].dt.month.astype(str)
    data["IsHoliday"] = data["datetime"].dt.date.apply(lambda x: x in fr_holidays)
    
    return data

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

# Has to be used after codify date!!!
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

def covid_19(data):
    date_ranges = [
        ("2020-10-30", "2020-12-15"),
        ("2021-03-20", "2021-05-19"),
#         ("2020-10-30", "2021-05-19")
    ]

    data["Covid-19"] = 0 
    for start, end in date_ranges:
        data["Covid-19"] |= data["datetime"].between(start, end).astype(int)

    return data

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

# Deletes the First Week
def add_lag_and_rolling_features(data):
    data_lag_rolling = (
        data.groupby('counter_id')
        .apply(_add_lag_and_rolling_features_group)
        .reset_index(drop=True)
        .query(f"datetime > '{data["datetime"].min() + pd.offsets.Week()}'")
    )
    return data_lag_rolling


def get_X_y(data):
    data = data.drop(columns=["counter_id", "site_id", "site_name", 
                              "bike_count", "counter_installation_date", 
                              "coordinates", "counter_technical_id",
                              "latitude", "longitude", "datetime", "date"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array
