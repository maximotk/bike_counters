import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import series_long_to_dict
from skforecast.preprocessing import exog_long_to_dict

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
