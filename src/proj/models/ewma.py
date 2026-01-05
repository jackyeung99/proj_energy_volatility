# models/ewma.py
import pandas as pd
from .base import VolatilityModel


class EWMAVariance(VolatilityModel):
    """
    EWMA variance model.

    If input_type == "returns":
        sigma_t^2 = EWMA(r_{t-1}^2)

    If input_type == "variance":
        sigma_t^2 = EWMA(v_{t-1})
    """

    def __init__(
        self,
        lam: float,
        input_type: str = "returns"  # "returns" or "variance"
    ):
        if input_type not in {"returns", "variance"}:
            raise ValueError("input_type must be 'returns' or 'variance'.")

        self.lam = lam
        self.column = 'ret' if input_type == 'returns' else 'rv'
        self.input_type = input_type

    @property
    def name(self):
        src = "ret" if self.input_type == "returns" else "rv"
        return f"EWMA({src}, λ={self.lam})"

    def _variance_proxy(self, data: pd.DataFrame) -> pd.Series:
        x = data[self.column]

        if self.input_type == "returns":
            return x ** 2
        else:  # "variance"
            return x

    def fit(self, data: pd.DataFrame):
        v = self._variance_proxy(data)

        self.fitted_variance_ = (
            v.shift(1)
             .ewm(alpha=1 - self.lam, adjust=False)
             .mean()
        )

        return self

    def forecast(self, data: pd.DataFrame) -> pd.Series:
        v = self._variance_proxy(data)

        f = (
            v.shift(1)
             .ewm(alpha=1 - self.lam, adjust=False)
             .mean()
        )

        # return aligned 1-step-ahead forecast
        return f.iloc[-1:]
