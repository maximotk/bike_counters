# 🚲 Predicting Cyclist Traffic in Paris

This repository implements **Predicting Cyclist Traffic in Paris**, a university project at École Polytechnique applying **machine learning** to forecast hourly bike counts at stations across Paris.  
It combines **gradient boosting** and **autoregressive** modeling approaches with **feature engineering** on weather, temporal, and COVID-related data.

👥 **Developed by:** [@maximotk](https://github.com/maximotk) & [@TheOneJoao](https://github.com/TheOneJoao)

---

## 📦 Installation & Requirements

To install requirements locally:

```bash
pip install -r requirements.txt
```

Alternatively, you can use **Poetry** for reproducible dependency management:

```bash
poetry install
poetry run streamlit run src/app.py
```

Or build and run the **Docker** image (recommended for deployment):

```bash
docker build -t bike-counters .
docker run -p 8501:8501 bike-counters
```

To pull the prebuilt image directly from **Docker Hub**:

```bash
docker pull maximotk/bike-counters:latest
docker run -d -p 8501:8501 maximotk/bike-counters:latest
```

After starting the container, open your browser at **http://localhost:8501**.

---

## 📈 Project Overview

This study investigates how to **predict hourly cyclist counts** across **56 counters** and **30 stations** in Paris.  
The training data covers **September 2020 – September 2021**, and models are evaluated on data from **September – October 2021**.

The dashboard supports interactive exploration of:

- 🗺️ **Geospatial overview** of counter locations across the city  
- 🧠 **Model performance** (global cumulative RMSE evolution)  
- 📍 **Station-level predictions** (actual vs. predicted values)  

---

## 🏋️ Training / Preprocessing Pipeline

The prediction workflow follows these main steps:

1. **Data Loading & Cleaning**  
   - Hourly counter data from 56 stations  
   - Outlier removal (malfunctioning counters with full-day zero counts)

2. **Feature Engineering**  
   - **Datetime features**: hour, day, weekday, month, weekend, holiday  
   - **COVID-19 features**: restriction periods and Oxford Stringency Index  
   - **Weather features**: temperature, precipitation, wind, humidity, etc.  
   - **Lag / Rolling statistics**: previous-hour/day/week means and std.  

3. **Modeling Approaches**  
   - **Gradient Boosting (XGBoost)** – captures non-linear interactions  
   - **Autoregressive model (skforecast)** – exploits temporal dependencies  

4. **Evaluation Metrics**  
   - Primary metric: **Root Mean Squared Error (RMSE)**  
   - Cumulative RMSE tracked over the test period  

---

## 🚀 Results

| Model | Description | Test RMSE (↓) | Notes |
|:------|:-------------|:-------------:|:------|
| **XGBoost** | Gradient boosting with temporal & COVID features | **≈ 0.63** | Best generalization |
| **Autoregressive** | Time-series model with exogenous variables | ≈ 0.66 | Propagated error over horizon |

Key observations:
- **Daily and weekly seasonality** captured effectively by datetime and lag features.  
- **Weather features** improved validation scores but not test performance → excluded from final model.  
- **XGBoost** achieved stable results without overfitting, balancing interpretability and predictive strength.

---

## 🎮 Running the Dashboard

To launch the interactive Streamlit dashboard locally:

```bash
streamlit run src/app.py
```

Or via Poetry:

```bash
poetry run streamlit run src/app.py
```

Once running, you can:

- 🗺️ Explore counter locations across Paris  
- ⚙️ Configure preprocessing and model settings in the sidebar  
- ▶️ Run the full modeling pipeline and visualize:
  - **Global error evolution**
  - **Station-level prediction curves**

All results are computed dynamically and stored in the **Streamlit session state** for interactivity.

---

## 📥 Data

- Data source: **Paris open bicycle counter dataset** (hourly counts from 56 stations).  
- Weather data merged from **Météo-France** hourly records.  
- COVID-19 indicators derived from the **Oxford Government Response Tracker**.  
- All processed data used in the dashboard is stored under `data/train.parquet`.  

---

## 📜 License

This repository is part of a **university research project** and is shared for **academic evaluation purposes only**.  
It is **not open for external contributions** at this time.
