import pandas as pd

def add_weather(df: pd.DataFrame, weather_path: str = "data/external_data.csv") -> pd.DataFrame:
    """
    Merge external weather data with the main dataset based on datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least a 'datetime' column.
    weather_path : str, optional
        Path to the external weather CSV file (default: "data/external_data.csv").

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with weather features:
        ['t', 'rr1', 'u', 'ht_neige', 'raf10', 'ff', 'ww', 'etat_sol', 'tend'].
    """
    weather = pd.read_csv(weather_path).drop_duplicates()
    weather = weather[["date", "t", "rr1", "u", "ht_neige", "raf10", "ff", "ww", "etat_sol", "tend"]]

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    weather["date"] = pd.to_datetime(weather["date"])

    df["original_index"] = df.index

    df_sorted = df.sort_values("datetime")
    weather_sorted = weather.sort_values("date")

    df_merged = pd.merge_asof(
        df_sorted,
        weather_sorted,
        left_on="datetime",
        right_on="date",
        direction="backward",
        suffixes=("", "_weather"),
    )

    df_merged = df_merged.sort_values("original_index").set_index("original_index")

    df_merged = df_merged.drop(columns=["date_weather"])

    return df_merged
