from xgboost import XGBRegressor

from features.covid_features import add_covid_stringency_index
from features.weather_features import add_weather
from features.date_features import add_date_features
from features.preprocessing import remove_outliers, handle_missing_values, cyclic_transform, one_hot_encode

from typing import Optional
import pandas as pd

class XGBoostModel:
    """
    XGBoostModel wrapper for a tuned XGBoost regressor.

    Parameters
    ----------
    random_state : Optional[int]
        Random seed for reproducibility. If None, uses the default from XGBoost.
    **kwargs
        Extra keyword arguments passed to XGBRegressor (will override defaults).

    Attributes
    ----------
    model : XGBRegressor
        The underlying XGBoost regressor instance.
    """

    def __init__(self, random_state: Optional[int] = 1, **kwargs) -> None:
        # default, tuned hyperparameters (kept from original)
        params = dict(
            colsample_bytree=0.8494252738248523,
            gamma=0.8835608079221302,
            learning_rate=0.12825147053070918,
            max_depth=8,
            n_estimators=428,
            reg_alpha=5.479087800903766,
            reg_lambda=6.995216197905481,
            subsample=0.6983244655616523,
            random_state=random_state
        )
        # user-supplied kwargs override defaults
        params.update(kwargs)

        self.model: XGBRegressor = XGBRegressor(**params)

    def preprocess(
        self,
        data: pd.DataFrame,
        base_path: str,
        remove_outliers_flag: bool = True,
        add_covid_flag: bool = True,
        add_weather_flag: bool = True,
        handle_missing_flag: bool = True,
        missing_method: str = "linear"
    ) -> pd.DataFrame:
        """
        Preprocess the input data with optional steps.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input dataframe (must contain the columns referenced above).
        base_path : str
            Base path used to locate external CSVs (e.g., Covid index, weather data).
        remove_outliers_flag : bool, default=True
            Whether to remove outliers.
        add_covid_flag : bool, default=True
            Whether to add COVID stringency index.
        add_weather_flag : bool, default=True
            Whether to add weather data.
        handle_missing_flag : bool, default=True
            Whether to handle missing values.
        missing_method : str, default="linear"
            Method for handling missing values.

        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe ready for `.fit()` / `.predict()`.
        """
        mdata = data.copy()

        mdata = mdata.pipe(add_date_features)
        print(data.dtypes)

        if remove_outliers_flag:
            mdata = mdata.pipe(remove_outliers)

        if add_covid_flag:
            mdata = mdata.pipe(
                add_covid_stringency_index, 
                path=base_path + "external_data/Covid_19_Index.csv"
            )

        if add_weather_flag:
            mdata = mdata.pipe(
                add_weather, 
                path=base_path + "external_data/weather_data.csv"
            )

        if handle_missing_flag:
            mdata = mdata.pipe(handle_missing_values, method=missing_method)

        mdata = (
            mdata
            .pipe(cyclic_transform, col="hour", period=24)
            .assign(counter_name_dup=lambda x: x["counter_name"])
            .pipe(one_hot_encode, cols=["counter_name", "year", "month", "day", "day_of_week"])
            .rename(columns={"counter_name_dup": "counter_name"})
            .drop(
                columns=[
                    "counter_id", "site_id", "site_name", "counter_installation_date",
                    "coordinates", "counter_technical_id",
                    "latitude", "longitude", "date"
                ],
                errors="ignore"
            )
            .pipe(lambda df: df.astype({col: float for col in df.select_dtypes(include="int").columns}))
            .pipe(lambda df: df.astype({col: float for col in df.select_dtypes(include="bool").columns}))
        )

        return mdata


    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs) -> None:
        """
        Fit the XGBoost regressor.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (preprocessed).
        y : pd.Series or array-like
            Target values aligned with X.
        **fit_kwargs :
            Additional keyword args forwarded to `XGBRegressor.fit`.
        """
        # XGBoost can accept DataFrame / Series directly
        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the fitted XGBoost model.

        Parameters
        ----------
        X : pd.DataFrame
            Preprocessed feature matrix.

        Returns
        -------
        pd.Series
            Predicted values, aligned with X.index.
        """
        preds = self.model.predict(X)
        # return as pandas Series to preserve index alignment
        return pd.Series(preds, name="prediction")
