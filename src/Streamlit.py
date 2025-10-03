# app.py
import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime

# import your model wrappers (paths must be correct relative to this file)
from models.xgboost import XGBoostModel
from models.autoregressive import AutoregressiveModel

st.set_page_config(layout="wide", page_title="Model Error Explorer")

# -------------------------
# Helpers & caching
# -------------------------
@st.cache_data
def load_data(parquet_path: str) -> pd.DataFrame:
    """Load the training parquet file (cached)."""
    df = pd.read_parquet(parquet_path)
    # ensure datetime column is datetime dtype
    if df.get("datetime", None) is not None:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def build_cumulative_error_series(merged_df, time_col="datetime", actual_col="log_bike_count", pred_col="log_bike_count_pred"):
    """
    Returns a DataFrame with index=time sorted ascending and a column 'rmse_cumulative'
    which is the RMSE computed across all observations up to each timestamp.
    """
    # Sort by time
    merged_df = merged_df.sort_values(time_col)
    
    cumulative_rmses = []
    sq_error_sum = 0
    total_count = 0
    for t, group in merged_df.groupby(time_col):
        errors = group[actual_col] - group[pred_col]
        sq_error_sum += (errors ** 2).sum()
        total_count += len(errors)
        cumulative_rmse = (sq_error_sum / total_count) ** 0.5
        cumulative_rmses.append((t, cumulative_rmse))
    
    return pd.DataFrame(cumulative_rmses, columns=[time_col, "rmse_cumulative"])

def build_counter_error_series(merged_df, counter_name, time_col="datetime", actual_col="log_bike_count", pred_col="log_bike_count_pred"):
    df = merged_df[merged_df["counter_name"] == counter_name].sort_values(time_col)
    df = df.assign(error=np.abs(df[actual_col] - df[pred_col]))
    # could return rolling RMSE or absolute error over time
    df["rmse_rolling_24"] = df["error"].rolling(window=24, min_periods=1).apply(lambda x: sqrt(np.mean(x**2)))
    return df[[time_col, "error", "rmse_rolling_24"]]
import plotly.graph_objects as go

def make_plotly_cumulative_animation(x, y, title="Evolution", y_label="RMSE"):
    """
    Build a Plotly figure with frames that animate drawing the line sequentially.
    """
    frames = []
    for i in range(0, len(x), 4):
        frames.append(go.Frame(
            data=[go.Scatter(x=x[: i + 1], y=y[: i + 1], mode="lines+markers")],
            name=str(i)
        ))

    # initial trace (first point)
    fig = go.Figure(
        data=[go.Scatter(x=[x.iloc[0]], y=[y.iloc[0]], mode="lines+markers")],
        frames=frames
    )
    fig.update_layout(
        title=title,
        xaxis_title="Datetime",
        yaxis_title=y_label,
        # Fix x-axis to full range
        xaxis=dict(range=[x.min(), x.max()]),
        yaxis=dict(range=[0, y.max()]),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "y": 1.05,
            "x": 1.02,
            "xanchor": "right",
            "yanchor": "top",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 1, "redraw": True},  # faster
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                },
            ],
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate"}],
                    "label": i,
                    "method": "animate",
                } for i, f in enumerate(frames)
            ],
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "len": 0.8
        }],
        margin=dict(l=40, r=20, t=60, b=40),
        height=420
    )
    return fig


base_path = ""
parquet_path = base_path + "data/train.parquet"
end_train = "2021-08-09 23:00:00"

# -------------------------
# UI: Top menu (no sidebar)
# -------------------------
st.title("Model & Feature Engineering Explorer")
st.markdown("Choose model and feature options, then run to compute and visualise error evolution.")

# ---- Layout: Model left, Features right ----
col_model, col_features = st.columns([1, 3])

# ---- Model (left, vertical) ----
with col_model:
    st.markdown("**Model**")
    model_choice = st.radio(
        "Model", 
        ["XGBoost", "Autoregressive"], 
        index=0, 
        label_visibility="collapsed"
    )

# ---- Features (right, horizontal) ----
with col_features:
    st.markdown("**Feature Engineering**")
    
    # horizontal checkboxes using columns
    cb1, cb2, cb3 = st.columns(3)
    with cb1:
        remove_outliers_flag = st.checkbox("Remove outliers", value=True)
    with cb2:
        add_covid_flag = st.checkbox("Add COVID index", value=True)
    with cb3:
        add_weather_flag = st.checkbox("Add weather features", value=True)
    
    # Fill missing + method in one row
    fill_col1, fill_col2, _ = st.columns([1, 1, 1])
    with fill_col1:
        handle_missing_flag = st.checkbox("Fill missing values", value=True)
    with fill_col2:
        if handle_missing_flag:
            missing_method = st.selectbox("Filling method", ["linear", "quadratic"], index=0)
        else:
            missing_method = None


# ---- Run button ----
run_button = st.button("Run pipeline & Visualize")

# -------------------------
# Main pipeline / visualization
# -------------------------
if run_button:
    with st.spinner("Loading data and running pipeline..."):
        # load data
        data = load_data(parquet_path)

        # instantiate model and preprocess accordingly
        if model_choice == "XGBoost":
            model = XGBoostModel()  # you can pass overrides if needed
            preproc = model.preprocess(
                data,
                base_path=base_path,
                remove_outliers_flag=remove_outliers_flag,
                add_covid_flag=add_covid_flag,
                add_weather_flag=add_weather_flag,
                handle_missing_flag=handle_missing_flag,
                missing_method=missing_method
            )
            mod_data = preproc

            # split
            mod_data["datetime"] = pd.to_datetime(mod_data["datetime"])
            train_data = mod_data[mod_data["datetime"] <= pd.to_datetime(end_train)].copy()
            test_data = mod_data[mod_data["datetime"] > pd.to_datetime(end_train)].copy()

            X_train = train_data.drop(columns=["log_bike_count", "datetime", "counter_name"], errors="ignore")
            y_train = train_data["log_bike_count"]
            model.fit(X=X_train, y=y_train)

            X_test = test_data.drop(columns=["log_bike_count", "datetime", "counter_name"], errors="ignore")
            preds = model.predict(X_test)  # pandas Series
            # attach predictions to test dataframe
            preds_df = test_data[["counter_name", "datetime"]].reset_index(drop=True).copy()
            preds_df["log_bike_count_pred"] = preds.values

            merged = test_data.reset_index(drop=True).merge(
                preds_df, on=["counter_name", "datetime"], how="left", suffixes=("_actual", "_pred")
            )
            # ensure names consistent
            merged.rename(columns={"log_bike_count": "log_bike_count_actual"}, inplace=True)
            # unify actual column name used below
            merged["log_bike_count"] = merged["log_bike_count_actual"]

        else:  # Autoregressive
            # configure AR model with correct time/space/endog names used in your data
            time_col = "datetime"
            space_col = "counter_name"
            endog_col = "log_bike_count"
            ar = AutoregressiveModel(time=time_col, frequency="H", space=space_col, endog=endog_col)

            mod_data = ar.preprocess(
                data,
                base_path=base_path,
                remove_outliers_flag=remove_outliers_flag,
                add_covid_flag=add_covid_flag,
                add_weather_flag=add_weather_flag,
                handle_missing_flag=handle_missing_flag,
                missing_method=missing_method
            )
            mod_data["datetime"] = pd.to_datetime(mod_data["datetime"])
            train_data = mod_data[mod_data["datetime"] <= pd.to_datetime(end_train)].copy()
            test_data = mod_data[mod_data["datetime"] > pd.to_datetime(end_train)].copy()

            # fit expects exog X and y
            X_train = train_data.drop(columns=["log_bike_count"], errors="ignore")
            y_train = train_data["log_bike_count"]
            ar.fit(X_train, y_train)

            preds_long = ar.predict(test_data.drop(columns=["log_bike_count"], errors="ignore"), steps=len(test_data["datetime"].unique()))
            # predict returns long-format with columns [datetime, counter_name, log_bike_count]
            preds_long = preds_long.rename(columns={endog_col: f"{endog_col}_pred"})
            # merge
            merged = test_data.merge(preds_long, on=["datetime", "counter_name"], how="left")
            merged.rename(columns={endog_col: "log_bike_count"}, inplace=True)
            merged["log_bike_count_pred"] = merged[f"{endog_col}_pred"]

        # Clean merged dataframe (drop rows with missing preds)
        merged = merged.dropna(subset=["log_bike_count_pred", "log_bike_count"]).copy()

        # compute instant & cumulative RMSE across all counters per timestamp
        cum_err_df = build_cumulative_error_series(merged, time_col="datetime", actual_col="log_bike_count", pred_col="log_bike_count_pred")

        # compute final overall measure
        final_overall_rmse = rmse(merged["log_bike_count"], merged["log_bike_count_pred"])

    # Display results and interactive charts
    st.markdown("### Overall error evolution (across all counters)")
    st.markdown(f"**Final overall RMSE (test):** {final_overall_rmse:.4f}")

    # animated overall RMSE plot (we'll animate the cumulative rmse series)
    fig_overall = make_plotly_cumulative_animation(cum_err_df["datetime"], cum_err_df["rmse_cumulative"], title="Overall RMSE (cumulative)", y_label="RMSE")
    st.plotly_chart(fig_overall, use_container_width=True)
