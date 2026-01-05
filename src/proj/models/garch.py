# models/garch.py
import pandas as pd
from arch import arch_model
from .base import VolatilityModel

class GARCHModel(VolatilityModel):
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q

    @property
    def name(self):
        return f"GARCH({self.p},{self.q})"

    def fit(self, data):
        r = data["ret_idio"]
        self.res = arch_model(
            r, mean="zero", vol="GARCH",
            p=self.p, q=self.q, dist="t"
        ).fit(disp="off")
        return self

    def forecast(self, data):
        f = self.res.forecast(horizon=1)
        return f.variance["h.1"]
