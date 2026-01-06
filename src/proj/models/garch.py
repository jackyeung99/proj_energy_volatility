# models/garch.py
import pandas as pd
from arch import arch_model
from .base import VolatilityModel

class GARCHModel(VolatilityModel):
    def __init__(self, p=1, q=1, dist="t"):
        self.p = p
        self.q = q
        self.dist = dist

    @property
    def name(self):
        return f"GARCH({self.p},{self.q})"

    def fit(self, data):
        r = data["ret"]
        self.res = arch_model(
            r, mean="zero", vol="GARCH",
            p=self.p, q=self.q, dist=self.dist
        ).fit(disp="off")
        self.fitted_variance_ = (self.res.conditional_volatility ** 2)
        self.fitted_variance_.index = data.index  # ensure aligned
        return self

    def forecast(self, data):
        f = self.res.forecast(horizon=1)
        return f.variance["h.1"]
