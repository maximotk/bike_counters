import pandas as pd
import numpy as np
import holidays

_target_column_name = "log_bike_count"

# Why only this? Why not month and day as well?
def codify_date(data):
    fr_holidays = holidays.France()

    data["datetime"] = data["date"]
    data["date"] = data["datetime"].dt.date
    data["hour"] = data["datetime"].dt.hour
    data["weekday"] = data["datetime"].dt.weekday
    data["daymonth"] = data["datetime"].dt.strftime('%d') + "_" + data["datetime"].dt.month.astype(str)
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

def get_X_y(data):
    data = data.drop(columns=["counter_id", "site_id", "site_name", 
                              "bike_count", "counter_installation_date", 
                              "coordinates", "counter_technical_id",
                              "latitude", "longitude", "datetime", "date"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array