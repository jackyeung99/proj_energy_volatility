# models/har_rv.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import VolatilityModel

class HARRV(VolatilityModel):
    def __init__(self, x_cols=None):
        self.x_cols = x_cols or []
        self.model = None

    @property
    def name(self):
        return "HAR-RV" if not self.x_cols else f"HAR-RV-X({len(self.x_cols)})"

    def _X_fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """Contemporaneous design matrix (for in-sample fitted values)."""
        rv = data["rv"]
        X = pd.DataFrame(index=data.index)
        X["d"] = np.log(rv.shift(1))
        X["w"] = np.log(rv.shift(1).rolling(5).mean())
        X["m"] = np.log(rv.shift(1).rolling(22).mean())

        for col in self.x_cols:
            # You can keep X lagged even in-sample if you want interpretability without simultaneity.
            # If you truly want max in-sample fit, remove shift here too.
            X[col] = data[col].shift(1)

        return sm.add_constant(X)

    def _X_forecast(self, data: pd.DataFrame) -> pd.DataFrame:
        """Lagged design matrix (for 1-step-ahead forecasting)."""
        rv = data["rv"]
        X = pd.DataFrame(index=data.index)
        X["d"] = np.log(rv.shift(1))
        X["w"] = np.log(rv.shift(1).rolling(5).mean())
        X["m"] = np.log(rv.shift(1).rolling(22).mean())

        for col in self.x_cols:
            X[col] = data[col].shift(1)

        return sm.add_constant(X)

    def fit(self, data: pd.DataFrame):
        y = np.log(data["rv"])

        X = self._X_fit(data)
        self.model = sm.OLS(y, X, missing="drop").fit()

        # In-sample fitted variance series (aligned to index)
        fitted_log = self.model.fittedvalues
        self.fitted_variance_ = np.exp(fitted_log).reindex(data.index)

        return self

    def forecast(self, data: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Call fit() before forecast().")

        Xf = self._X_forecast(data)
        return np.exp(Xf @ self.model.params).iloc[-1:]
