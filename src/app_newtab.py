# app_singleview.py
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
import folium
import plotly.express as px


from app_functions.data_loading import load_data
from app_functions.metrics import rmse, build_cumulative_error_series
from app_functions.plots import make_plotly_cumulative_animation, make_station_prediction_plot
from models.xgboost import XGBoostModel
from models.autoregressive import AutoregressiveModel


# -------------------------
# Page Config
# -------------------------
st.set_page_config(layout="wide", page_title="🚲 Bike Counter Error Explorer")

base_path = ""
parquet_path = base_path + "data/train.parquet"
end_train = "2021-08-09 23:00:00"


# -------------------------
# Title & Intro
# -------------------------
st.title("🚲 Bike Counter Error Explorer")
st.markdown("""
Welcome to the **Bike Counter Error Explorer** —  
an interactive dashboard to explore and visualize how predictive models perform  
on bicycle traffic counters across the city.

Use the sidebar to choose model options and feature-engineering steps,  
then hit **▶️ Run pipeline & Visualize** to train models and view predictions.
""")


# -------------------------
# Sidebar for options
# -------------------------
st.sidebar.header("⚙️ Model & Feature Configuration")
st.sidebar.markdown("Select your model setup:")

model_choice = st.sidebar.radio("Model", ["XGBoost", "Autoregressive"], index=0)

st.sidebar.markdown("**Feature Engineering Options**")
remove_outliers_flag = st.sidebar.checkbox("Remove outliers", value=True)
add_covid_flag = st.sidebar.checkbox("Add COVID index", value=True)
add_weather_flag = st.sidebar.checkbox("Add weather features", value=True)

handle_missing_flag = st.sidebar.checkbox("Fill missing values", value=True)
missing_method = st.sidebar.selectbox(
    "Filling method", ["linear", "quadratic"], index=0
) if handle_missing_flag else None

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶️ Run pipeline & Visualize")


# -------------------------
# Always Show: Overview Map
# -------------------------
st.subheader("🗺️ Counter Locations Overview")
st.markdown("""
Below you can explore all **bike counting stations** included in the dataset.  
Each blue marker represents one counter — click it to see the station’s name.
""")

try:
    data = load_data(parquet_path)

    # Prepare coordinates
    map_data = (
        data[["counter_name", "latitude", "longitude"]]
        .drop_duplicates("counter_name")
        .dropna(subset=["latitude", "longitude"])
    )

    m = folium.Map(location=map_data[["latitude", "longitude"]].mean().values.tolist(), zoom_start=12)
    for _, row in map_data.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=row["counter_name"],
            icon=folium.Icon(color="blue", icon="bicycle", prefix="fa")
        ).add_to(m)

    st_folium(m, width=1000, height=450)
    st.caption(f"Showing {len(map_data)} counter locations on the map.")
except Exception as e:
    st.warning(f"⚠️ Map could not be displayed: {e}")


# -------------------------
# Always Show: Explanation (before running)
# -------------------------
st.markdown("---")
st.header("🔮 Model Training & Prediction")
st.markdown("""
When you press **Run pipeline & Visualize**, the following steps are performed:

1. **Preprocessing** of training data with your selected feature-engineering options  
2. **Model training** on data up to August 2021  
3. **Prediction** for the remaining test period  
4. **Evaluation** of model performance via cumulative RMSE  

Once finished, the dashboard will display:
- 📈 **Global Error Evolution** (cumulative RMSE over time)  
- 📊 **Station-Level Predictions** (actual vs predicted values per counter)
""")


# -------------------------
# Run pipeline
# -------------------------
if run_button:
    st.markdown("---")
    with st.spinner("Running pipeline... this may take a few moments ⏳"):

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

        st.session_state["merged"] = merged
        st.session_state["final_overall_rmse"] = final_overall_rmse
        st.session_state["cum_err_df"] = cum_err_df


# -------------------------
# Visualization (if model was run)
# -------------------------
if "merged" in st.session_state:
    merged = st.session_state["merged"]
    cum_err_df = st.session_state["cum_err_df"]
    final_overall_rmse = st.session_state["final_overall_rmse"]

    st.markdown("---")
    st.header("📈 Model Performance Visualization")

    # --- Global RMSE Evolution ---
    st.subheader("Global Error Evolution")
    st.markdown("This plot shows how the **cumulative RMSE** evolved over time during the test period.")
    st.metric("Final overall RMSE (test)", f"{final_overall_rmse:.4f}")

    fig_overall = make_plotly_cumulative_animation(
        cum_err_df["datetime"], cum_err_df["rmse_cumulative"],
        title="Overall RMSE (Cumulative)", y_label="RMSE"
    )
    st.plotly_chart(fig_overall, use_container_width=True)

    # --- Station-level Analysis ---
    st.subheader("Station-Level Predictions")
    st.markdown("Select an individual counter below to inspect the model’s predictions vs. actual values.")

    station = st.selectbox("Select a counter", merged["counter_name"].unique())
    station_data = merged[merged["counter_name"] == station].copy()
    fig_line = make_station_prediction_plot(station_data, station)
    st.plotly_chart(fig_line, use_container_width=True)

