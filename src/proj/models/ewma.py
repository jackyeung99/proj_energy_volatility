# models/ewma.py
import pandas as pd
from .base import VolatilityModel

class EWMAVariance(VolatilityModel):
    def __init__(self, lam: float, column: str):
        self.lam = lam
        self.column = column

    @property
    def name(self):
        return f"EWMA(lam={self.lam})"

    def fit(self, data):
        x = data[self.column]
        self.fitted_variance_ = (
            x.shift(1)
            .ewm(alpha=1 - self.lam, adjust=False)
            .mean()
        )
        return self

    def forecast(self, data):
        x = data[self.column]
        return (
            x.shift(1)
             .ewm(alpha=1 - self.lam, adjust=False)
             .mean()
             .iloc[-1:]
        ) 
