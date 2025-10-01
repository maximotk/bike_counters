import pandas as pd
from pathlib import Path

def add_covid_lockdown_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary COVID-19 lockdown flag to the dataset.

    Lockdown periods considered:
    - 2020-10-30 to 2020-12-15
    - 2021-03-20 to 2021-05-19

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'datetime' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'covid_lockdown' column (0 = no lockdown, 1 = lockdown).
    """
    lockdown_ranges = [
        ("2020-10-30", "2020-12-15"),
        ("2021-03-20", "2021-05-19"),
    ]

    df = df.copy()
    df["covid_lockdown"] = 0
    for start, end in lockdown_ranges:
        df["covid_lockdown"] |= df["datetime"].between(start, end).astype(int)

    return df


def add_covid_stringency_index(df: pd.DataFrame, index_path: str = "data/Covid_19_Index.csv") -> pd.DataFrame:
    """
    Merge the Oxford COVID-19 Government Response Tracker stringency index for France.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'date' column.
    index_path : str, optional
        Path to the CSV file containing the stringency index data (default: "data/Covid_19_Index.csv").

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with the 'StringencyIndex_Average' column.

    Notes
    -----
    Source: Thomas Hale, Noam Angrist, Rafael Goldszmidt, Beatriz Kira, Anna Petherick,
    Toby Phillips, Samuel Webster, Emily Cameron-Blake, Laura Hallas, Saptarshi Majumdar,
    and Helen Tatlow. (2021). “A global panel database of pandemic policies (Oxford
    COVID-19 Government Response Tracker).” Nature Human Behaviour.
    https://doi.org/10.1038/s41562-021-01079-8
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    min_date, max_date = df["date"].min(), df["date"].max()

    covid_index = pd.read_csv(index_path)
    covid_index["date"] = pd.to_datetime(covid_index["Date"], format="%Y%m%d")

    covid_index = (
        covid_index
        .query("CountryName == 'France'")
        .query("date >= @min_date and date <= @max_date")
        [["date", "StringencyIndex_Average"]]
    )

    df = df.merge(covid_index, on="date", how="left")
    return df
