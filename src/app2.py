# app_alternative.py
import streamlit as st
import pandas as pd
from datetime import datetime

# Utils
from utils.data_utils import load_data
from utils.metrics import rmse, build_cumulative_error_series
from utils.plots import make_plotly_cumulative_animation

# Models
from models.xgboost import XGBoostModel
from models.autoregressive import AutoregressiveModel


# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(layout="wide", page_title="🚲 Bike Counter Model Explorer")

base_path = ""
parquet_path = base_path + "data/train.parquet"
end_train = "2021-08-09 23:00:00"


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Configuration")

# Model selection
model_choice = st.sidebar.radio(
    "Select a model",
    ["XGBoost", "Autoregressive"],
    index=0
)

# Feature engineering flags
st.sidebar.subheader("🛠 Feature Engineering")
remove_outliers_flag = st.sidebar.checkbox("Remove outliers", value=True)
add_covid_flag = st.sidebar.checkbox("Add COVID index", value=True)
add_weather_flag = st.sidebar.checkbox("Add weather features", value=True)

# Missing value handling
handle_missing_flag = st.sidebar.checkbox("Fill missing values", value=True)
missing_method = st.sidebar.selectbox(
    "Filling method",
    ["linear", "quadratic"],
    index=0
) if handle_missing_flag else None

# Run pipeline button
run_button = st.sidebar.button("🚀 Run Pipeline & Visualize")


# -------------------------
# Main page content
# -------------------------
st.title("🚲 Bike Counter Model & Feature Explorer")

st.markdown(
    """
    Welcome to the **Bike Counter Dashboard**!  
    Here you can experiment with different **models** and **feature engineering options**  
    to see how they affect prediction accuracy over time.  

    👉 Configure your options in the **sidebar** (left),  
    then press **🚀 Run Pipeline & Visualize** to start the analysis.
    """
)

# -------------------------
# Pipeline & Visualization
# -------------------------
if run_button:
    with st.spinner("Loading data and running pipeline..."):
        # Load data
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

            # Split train/test
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

    # -------------------------
    # Display results
    # -------------------------
    st.success("✅ Pipeline finished successfully!")

    st.markdown("### 📈 Overall error evolution (across all counters)")
    st.metric(label="Final overall RMSE (test)", value=f"{final_overall_rmse:.4f}")

    fig_overall = make_plotly_cumulative_animation(
        cum_err_df["datetime"], cum_err_df["rmse_cumulative"],
        title="Overall RMSE (cumulative)", y_label="RMSE"
    )
    st.plotly_chart(fig_overall, use_container_width=True)

else:
    # Before running, show instructions
    st.info("👈 Configure options in the sidebar and click **Run Pipeline & Visualize** to see results.")
