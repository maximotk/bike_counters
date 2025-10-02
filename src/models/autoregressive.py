import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import series_long_to_dict
from skforecast.preprocessing import exog_long_to_dict

from features.covid_features import add_covid_stringency_index
from features.date_features import add_date_features
from features.preprocessing import remove_outliers, handle_missing_values, cyclic_transform, one_hot_encode
from features.weather_features import add_weather

class AutoregressiveModel:
    """
    AutoregressiveModel multi-series forecaster using recursive boosting.

    Parameters
    ----------
    time : str
        Name of the time column in the input data.
    frequency : str
        Frequency string (e.g., 'H' for hourly) used for time series indexing.
    space : str
        Name of the spatial identifier column (e.g., location, counter).
    endog : str
        Name of the endogenous variable (target) column.

    Attributes
    ----------
    forecaster : ForecasterRecursiveMultiSeries
        The fitted forecaster model.
    """

    def __init__(self, time: str, frequency: str, space: str, endog: str):
        self.time = time
        self.frequency = frequency
        self.space = space
        self.endog = endog
        self.forecaster = None

    def preprocess(self, data: pd.DataFrame, base_path: str) -> pd.DataFrame:
        """
        Preprocess the input data: add covid and weather data, remove outliers, handle missing values, 
        apply cyclic transformations, encode categorical variables, and drop unnecessary columns.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe to preprocess.
        base_path : str
            Base path.

        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe ready for model fitting.
        """
        data = data.copy()
        data = (
            data
            .pipe(add_date_features)
            .pipe(remove_outliers)
            .pipe(add_covid_stringency_index, path=base_path+"external_data/Covid_19_Index.csv")
            .pipe(add_weather, path=base_path+"external_data/weather_data.csv")
            .pipe(handle_missing_values, method="linear")
            .pipe(cyclic_transform, col="hour", period=24)
            .pipe(one_hot_encode, cols=["year", "month", "day", "day_of_week"])
            .drop(columns=["counter_id", "site_id", "site_name", "counter_installation_date", 
                                    "coordinates", "counter_technical_id",
                                    "latitude", "longitude", "date", "bike_count"])
            .pipe(lambda df: df.astype({col: float for col in df.select_dtypes(include="int").columns}))
            .pipe(lambda df: df.astype({col: float for col in df.select_dtypes(include="bool").columns}))
        )

        return data

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the autoregressive model.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous features, including at least:
            - time column
            - space column
        y : pd.Series
            Endogenous target variable aligned with X.
        """
        endog_series = pd.concat(
            [X[[self.time, self.space]], pd.DataFrame(y, columns=[self.endog])],
            axis=1
        )
        exog_series = X

        endog_dict = series_long_to_dict(
            data=endog_series,
            series_id=self.space,
            index=self.time,
            values=self.endog,
            freq=self.frequency
        )

        exog_dict = exog_long_to_dict(
            data=exog_series,
            series_id=self.space,
            index=self.time,
            freq=self.frequency
        )

        window_features = RollingFeatures(
            stats=['mean', 'mean'], 
            window_sizes=[24, 168]
        )

        forecaster = ForecasterRecursiveMultiSeries(
            regressor=HistGradientBoostingRegressor(random_state=123),
            lags=[1, 24, 168],
            window_features=window_features,
            encoding="ordinal",
            dropna_from_series=False
        )

        forecaster.fit(series=endog_dict, exog=exog_dict, suppress_warnings=True)
        self.forecaster = forecaster

    def predict(self, X: pd.DataFrame, steps: int = 1020) -> pd.DataFrame:
        """
        Predict future values for the endogenous variable.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous features with the same structure as used in `fit`.
        steps : int, optional
            Number of forecasting steps ahead (default: 1020).

        Returns
        -------
        pd.DataFrame
            Long-format dataframe with columns:
            - time
            - space
            - endog (predicted values)
        """
        exog_series = X
        exog_dict = exog_long_to_dict(
            data=exog_series,
            series_id=self.space,
            index=self.time,
            freq=self.frequency
        )

        predictions = self.forecaster.predict(steps=steps, exog=exog_dict)
        return (
            predictions.reset_index(names=self.time)
            .melt(id_vars=[self.time], var_name=self.space, value_name=self.endog)
        )
