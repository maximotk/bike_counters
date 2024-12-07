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

def get_X_y(data):
    data = data.drop(columns=["counter_id", "site_id", "site_name", 
                              "bike_count", "counter_installation_date", 
                              "coordinates", "counter_technical_id",
                              "latitude", "longitude", "datetime", "date"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array
