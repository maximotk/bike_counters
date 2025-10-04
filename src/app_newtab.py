# app_alternative.py
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_utils import load_data
from utils.metrics import rmse, build_cumulative_error_series
from utils.plots import make_plotly_cumulative_animation

from models.xgboost import XGBoostModel
from models.autoregressive import AutoregressiveModel


# -------------------------
# Page Config
# -------------------------
st.set_page_config(layout="wide", page_title="Bike Counter Error Explorer")

base_path = ""
parquet_path = base_path + "data/train.parquet"
end_train = "2021-08-09 23:00:00"


# -------------------------
# UI: Title
# -------------------------
st.title("🚲 Bike Counter Error Explorer")
st.markdown("Explore model error evolution globally and by station.")


# -------------------------
# Sidebar for options
# -------------------------
st.sidebar.header("⚙️ Options")

model_choice = st.sidebar.radio("Model", ["XGBoost", "Autoregressive"], index=0)

remove_outliers_flag = st.sidebar.checkbox("Remove outliers", value=True)
add_covid_flag = st.sidebar.checkbox("Add COVID index", value=True)
add_weather_flag = st.sidebar.checkbox("Add weather features", value=True)

handle_missing_flag = st.sidebar.checkbox("Fill missing values", value=True)
missing_method = st.sidebar.selectbox(
    "Filling method", ["linear", "quadratic"], index=0
) if handle_missing_flag else None

run_button = st.sidebar.button("▶️ Run pipeline & Visualize")


# -------------------------
# Run pipeline
# -------------------------
if run_button:
    with st.spinner("Loading data and running pipeline..."):
        # Load
        data = load_data(parquet_path)

        # -------------------------
        # Model-specific processing
        # -------------------------
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
            train_data = mod_data[mod_data["datetime"] <= pd.to_datetime(end_train)].copy()
            test_data = mod_data[mod_data["datetime"] > pd.to_datetime(end_train)].copy()

            X_train = train_data.drop(columns=["log_bike_count", "datetime", "counter_name"], errors="ignore")
            y_train = train_data["log_bike_count"]
            model.fit(X=X_train, y=y_train)

            X_test = test_data.drop(columns=["log_bike_count", "datetime", "counter_name"], errors="ignore")
            preds = model.predict(X_test)

            preds_df = test_data[["counter_name", "datetime"]].reset_index(drop=True).copy()
            preds_df["log_bike_count_pred"] = preds.values

            merged = test_data.reset_index(drop=True).merge(
                preds_df, on=["counter_name", "datetime"], how="left", suffixes=("_actual", "_pred")
            )
            merged.rename(columns={"log_bike_count": "log_bike_count_actual"}, inplace=True)
            merged["log_bike_count"] = merged["log_bike_count_actual"]

        else:  # Autoregressive
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
            train_data = mod_data[mod_data["datetime"] <= pd.to_datetime(end_train)].copy()
            test_data = mod_data[mod_data["datetime"] > pd.to_datetime(end_train)].copy()

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

        # -------------------------
        # Evaluation
        # -------------------------
        merged = merged.dropna(subset=["log_bike_count_pred", "log_bike_count"]).copy()
        cum_err_df = build_cumulative_error_series(merged)
        final_overall_rmse = rmse(merged["log_bike_count"], merged["log_bike_count_pred"])

        # Save results into session state
        st.session_state["merged"] = merged
        st.session_state["final_overall_rmse"] = final_overall_rmse
        st.session_state["cum_err_df"] = cum_err_df


# -------------------------
# Tabs (only if results exist)
# -------------------------
if "merged" in st.session_state:
    merged = st.session_state["merged"]
    final_overall_rmse = st.session_state["final_overall_rmse"]
    cum_err_df = st.session_state["cum_err_df"]

    tab1, tab2 = st.tabs(["🌍 Global Performance", "📍 By Station Analysis"])

    # ---- Tab 1: Global ----
    with tab1:
        st.subheader("Overall error evolution (across all counters)")
        st.metric("Final overall RMSE (test)", f"{final_overall_rmse:.4f}")

        fig_overall = make_plotly_cumulative_animation(
            cum_err_df["datetime"], cum_err_df["rmse_cumulative"],
            title="Overall RMSE (cumulative)", y_label="RMSE"
        )
        st.plotly_chart(fig_overall, use_container_width=True)

        # Summary table
        st.subheader("Station Summary Table")
        summary = (
            merged.groupby("counter_name")
            .apply(lambda g: pd.Series({
                "RMSE": rmse(g["log_bike_count"], g["log_bike_count_pred"]),
                "MedianAbsError": (g["log_bike_count"] - g["log_bike_count_pred"]).abs().median(),
                "Samples": len(g)
            }))
            .reset_index()
            .sort_values("RMSE")
        )
        st.dataframe(summary, use_container_width=True)

    # ---- Tab 2: Station Analysis ----
    with tab2:
        st.subheader("Station-Level Performance")

        station = st.selectbox("Select a counter", merged["counter_name"].unique())
        station_data = merged[merged["counter_name"] == station].copy()

        # Actual vs Predicted
        fig_line = px.line(
            station_data, x="datetime",
            y=["log_bike_count", "log_bike_count_pred"],
            labels={"value": "Log Bike Count", "datetime": "Date"},
            title=f"Actual vs Predicted: {station}"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Rolling RMSE
        station_data["abs_error"] = (station_data["log_bike_count"] - station_data["log_bike_count_pred"]).abs()
        station_data["rmse_rolling_24"] = (
            station_data["abs_error"].rolling(24, min_periods=1).apply(lambda x: (x**2).mean()**0.5)
        )
        fig_rmse = px.line(
            station_data, x="datetime", y="rmse_rolling_24",
            labels={"rmse_rolling_24": "24h Rolling RMSE"},
            title=f"24h Rolling RMSE: {station}"
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
