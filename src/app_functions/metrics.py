import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def build_cumulative_error_series(merged_df, time_col="datetime", actual_col="log_bike_count", pred_col="log_bike_count_pred"):
    merged_df = merged_df.sort_values(time_col)
    cumulative_rmses, sq_error_sum, total_count = [], 0, 0
    for t, group in merged_df.groupby(time_col):
        errors = group[actual_col] - group[pred_col]
        sq_error_sum += (errors ** 2).sum()
        total_count += len(errors)
        cumulative_rmse = (sq_error_sum / total_count) ** 0.5
        cumulative_rmses.append((t, cumulative_rmse))
    return pd.DataFrame(cumulative_rmses, columns=[time_col, "rmse_cumulative"])
