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
        # EWMA has no parameters to fit
        return self

    def forecast(self, data):
        x = data[self.column]
        return (
            x.ewm(alpha=1 - self.lam, adjust=False)
             .mean()
        )
