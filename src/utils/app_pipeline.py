import pandas as pd
from utils.metrics import rmse, build_cumulative_error_series
from models.xgboost import XGBoostModel
from models.autoregressive import AutoregressiveModel

def run_pipeline(
    data: pd.DataFrame,
    model_choice: str,
    base_path: str,
    end_train: str,
    remove_outliers_flag: bool,
    add_covid_flag: bool,
    add_weather_flag: bool,
    handle_missing_flag: bool,
    missing_method: str,
):
    if model_choice == "XGBoost":
        model = XGBoostModel()
        mod_data = model.preprocess(
            data,
            base_path=base_path,
            remove_outliers_flag=remove_outliers_flag,
            add_covid_flag=add_covid_flag,
            add_weather_flag=add_weather_flag,
            handle_missing_flag=handle_missing_flag,
            missing_method=missing_method,
        )
        mod_data["datetime"] = pd.to_datetime(mod_data["datetime"])
        train_data = mod_data[mod_data["datetime"] <= pd.to_datetime(end_train)]
        test_data = mod_data[mod_data["datetime"] > pd.to_datetime(end_train)]

        X_train = train_data.drop(columns=["log_bike_count", "datetime", "counter_name"], errors="ignore")
        y_train = train_data["log_bike_count"]
        model.fit(X_train, y_train)

        X_test = test_data.drop(columns=["log_bike_count", "datetime", "counter_name"], errors="ignore")
        preds = model.predict(X_test)

        preds_df = test_data[["counter_name", "datetime"]].reset_index(drop=True).copy()
        preds_df["log_bike_count_pred"] = preds.values

        merged = test_data.reset_index(drop=True).merge(
            preds_df, on=["counter_name", "datetime"], how="left"
        )
        merged.rename(columns={"log_bike_count": "log_bike_count_actual"}, inplace=True)
        merged["log_bike_count"] = merged["log_bike_count_actual"]

    else:
        time_col, space_col, endog_col = "datetime", "counter_name", "log_bike_count"
        ar = AutoregressiveModel(time=time_col, frequency="H", space=space_col, endog=endog_col)
        mod_data = ar.preprocess(
            data,
            base_path=base_path,
            remove_outliers_flag=remove_outliers_flag,
            add_covid_flag=add_covid_flag,
            add_weather_flag=add_weather_flag,
            handle_missing_flag=handle_missing_flag,
            missing_method=missing_method,
        )
        mod_data["datetime"] = pd.to_datetime(mod_data["datetime"])
        train_data = mod_data[mod_data["datetime"] <= pd.to_datetime(end_train)]
        test_data = mod_data[mod_data["datetime"] > pd.to_datetime(end_train)]

        X_train = train_data.drop(columns=["log_bike_count"], errors="ignore")
        y_train = train_data["log_bike_count"]
        ar.fit(X_train, y_train)

        preds_long = ar.predict(
            test_data.drop(columns=["log_bike_count"], errors="ignore"),
            steps=len(test_data["datetime"].unique())
        )
        preds_long = preds_long.rename(columns={endog_col: f"{endog_col}_pred"})

        merged = test_data.merge(preds_long, on=["datetime", "counter_name"], how="left")
        merged.rename(columns={endog_col: "log_bike_count"}, inplace=True)
        merged["log_bike_count_pred"] = merged[f"{endog_col}_pred"]

    merged = merged.dropna(subset=["log_bike_count_pred", "log_bike_count"]).copy()
    cum_err_df = build_cumulative_error_series(merged)
    final_rmse = rmse(merged["log_bike_count"], merged["log_bike_count_pred"])
    return merged, cum_err_df, final_rmse
