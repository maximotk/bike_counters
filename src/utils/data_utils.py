import pandas as pd
import streamlit as st

@st.cache_data
def load_data(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if df.get("datetime", None) is not None:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df
