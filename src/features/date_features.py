import pandas as pd
import holidays


def add_date_features(df: pd.DataFrame, use_extended: bool = True) -> pd.DataFrame:
    """
    Add date and time related features to a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'date' or 'datetime' column.
    use_extended : bool, optional (default=True)
        - If True: Adds full set of features (year, month, day, weekday, is_weekend, holidays).
        - If False: Adds a simpler set of features (hour, weekday, day-month, holiday).

    Returns
    -------
    pd.DataFrame
        DataFrame with new date-related feature columns.
    """
    fr_holidays = holidays.France()
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["datetime"] = pd.to_datetime(df["date"])
    else:
        df["datetime"] = df["date"]

    df["date"] = df["datetime"].dt.date

    if use_extended:
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["day_of_week"] = df["datetime"].dt.dayofweek  # Monday=0
        df["hour"] = df["datetime"].dt.hour
        df["is_weekend"] = df["datetime"].dt.dayofweek >= 5
        df["IsHoliday"] = df["datetime"].dt.date.apply(lambda x: x in fr_holidays)

    else:
        df["hour"] = df["datetime"].dt.hour
        df["weekday"] = df["datetime"].dt.weekday
        df["daymonth"] = (
            df["datetime"].dt.strftime("%d") + "_" + df["datetime"].dt.month.astype(str)
        )
        df["IsHoliday"] = df["datetime"].dt.date.apply(lambda x: x in fr_holidays)

    return df
