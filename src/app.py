import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from app_functions.map_creation import make_counter_map
from app_functions.app_pipeline import run_pipeline
from app_functions.data_loading import load_data
from app_functions.plots import make_plotly_cumulative_animation, make_station_prediction_plot

st.set_page_config(layout="wide", page_title="🚲 Bike Counter Error Explorer")

base_path = ""
parquet_path = base_path + "data/train.parquet"
end_train = "2021-08-09 23:00:00"

# --- Title & Intro ---
st.title("🚲 Bike Counter Error Explorer")
st.markdown("""
Welcome to the **Bike Counter Error Explorer** —  
an interactive dashboard to explore and visualize predictive model performance  
on bicycle traffic counters across the city.
""")

# --- Sidebar ---
st.sidebar.header("⚙️ Model & Feature Configuration")
model_choice = st.sidebar.radio("Model", ["XGBoost", "Autoregressive"], index=0)
remove_outliers_flag = st.sidebar.checkbox("Remove outliers", value=True)
add_covid_flag = st.sidebar.checkbox("Add COVID index", value=True)
add_weather_flag = st.sidebar.checkbox("Add weather features", value=True)
handle_missing_flag = st.sidebar.checkbox("Fill missing values", value=True)
missing_method = st.sidebar.selectbox(
    "Filling method", ["linear", "quadratic"], index=0
) if handle_missing_flag else None
st.sidebar.markdown("---")
run_button = st.sidebar.button("▶️ Run pipeline & Visualize")

# --- Map ---
st.subheader("🗺️ Counter Locations Overview")
st.markdown("Explore all **bike counting stations** below — click a marker to view its name.")
data = load_data(parquet_path)
make_counter_map(data)

# --- Introduction ---
st.markdown("---")
st.header("🔮 Model Training & Prediction")
st.markdown("""
When you press **Run pipeline & Visualize**, the app:
1. Preprocesses training data based on your feature settings  
2. Trains your chosen model up to August 2021  
3. Predicts for the remaining period  
4. Evaluates model performance via cumulative RMSE  
""")

# --- Run pipeline ---
if run_button:
    with st.spinner("Running pipeline... this may take a few moments ⏳"):
        merged, cum_err_df, final_overall_rmse = run_pipeline(
            data=data,
            model_choice=model_choice,
            base_path=base_path,
            end_train=end_train,
            remove_outliers_flag=remove_outliers_flag,
            add_covid_flag=add_covid_flag,
            add_weather_flag=add_weather_flag,
            handle_missing_flag=handle_missing_flag,
            missing_method=missing_method,
        )
        st.session_state["merged"] = merged
        st.session_state["cum_err_df"] = cum_err_df
        st.session_state["final_overall_rmse"] = final_overall_rmse

# --- Visualization ---
if "merged" in st.session_state:
    merged = st.session_state["merged"]
    cum_err_df = st.session_state["cum_err_df"]
    final_overall_rmse = st.session_state["final_overall_rmse"]

    st.markdown("---")
    st.header("📈 Model Performance Visualization")

    st.subheader("Global Error Evolution")
    st.metric("Final overall RMSE (test)", f"{final_overall_rmse:.4f}")
    fig_overall = make_plotly_cumulative_animation(
        cum_err_df["datetime"], cum_err_df["rmse_cumulative"],
        title="Overall RMSE (Cumulative)", y_label="RMSE"
    )
    st.plotly_chart(fig_overall, use_container_width=True)

    st.subheader("Station-Level Predictions")
    st.markdown("Select a counter to inspect model predictions vs. actual values.")
    station = st.selectbox("Select a counter", merged["counter_name"].unique())
    station_data = merged[merged["counter_name"] == station].copy()
    fig_line = make_station_prediction_plot(station_data, station)
    st.plotly_chart(fig_line, use_container_width=True)
