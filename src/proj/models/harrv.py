# models/har_rv.py
import numpy as np
import statsmodels.api as sm
import pandas as pd
from .base import VolatilityModel

class HARRV(VolatilityModel):
    def __init__(self, x_cols=None):
        self.x_cols = x_cols or []

    @property
    def name(self):
        return "HAR-RV" if not self.x_cols else "HAR-RV-X"

    def fit(self, data):
        y = np.log(data["rv"])
        X = pd.DataFrame({
            "d": np.log(data["rv"]),
            "w": np.log(data["rv"].rolling(5).mean()),
            "m": np.log(data["rv"].rolling(22).mean()),
        })

        for col in self.x_cols:
            X[col] = data[col].shift(1)

        X = sm.add_constant(X)
        self.model = sm.OLS(y, X, missing="drop").fit()
        return self

    def forecast(self, data):
        Xf = pd.DataFrame({
            "d": np.log(data["rv"].shift(1)),
            "w": np.log(data["rv"].shift(1).rolling(5).mean()),
            "m": np.log(data["rv"].shift(1).rolling(22).mean()),
        })

        for col in self.x_cols:
            Xf[col] = data[col].shift(1)

        Xf = sm.add_constant(Xf)
        return np.exp(Xf @ self.model.params)
