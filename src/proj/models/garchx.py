# models/garch_x_proxy.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from .base import VolatilityModel

class GARCHProxyX(VolatilityModel):
    """
    Two-step model:
      Step 1: Fit GARCH on returns -> sigma2_hat
      Step 2: Predict log RV using log sigma2_hat, lagged log RV, and X (lagged)

    Forecast output: variance (RV forecast).
    """
    def __init__(self, p=1, q=1, dist="t", x_cols=None):
        self.p = p
        self.q = q
        self.dist = dist
        self.x_cols = x_cols or []
        self._garch_res = None
        self._ols_res = None
        self._ols_params = None

    @property
    def name(self):
        return f"GARCHProxyX({self.p},{self.q},{self.dist},X={len(self.x_cols)})"

    def fit(self, data: pd.DataFrame):
        r = data["ret_idio"]
        rv = data["rv_idio"]

        # Step 1: GARCH on returns
        self._garch_res = arch_model(
            r, mean="zero", vol="GARCH",
            p=self.p, q=self.q, dist=self.dist, rescale=True
        ).fit(disp="off")

        sigma2 = (self._garch_res.conditional_volatility ** 2)
        log_sigma2 = np.log(sigma2)

        # Step 2: OLS on log RV with lagged predictors
        df = pd.DataFrame({
            "y": np.log(rv),
            "log_sigma2": log_sigma2,
            "log_rv_lag1": np.log(rv.shift(1)),
        })

        for col in self.x_cols:
            df[col] = data[col].shift(1)

        df = df.dropna()
        X = sm.add_constant(df.drop(columns=["y"]))
        y = df["y"]

        self._ols_res = sm.OLS(y, X).fit()
        self._ols_params = self._ols_res.params
        return self

    def forecast(self, data: pd.DataFrame) -> pd.Series:
        # Recompute sigma2 using the fitted GARCH recursion requires re-filtering,
        # but simplest is to use in-sample conditional volatility from fit period.
        # For 1-step-ahead backtesting, pass data as train+test and refit in outer loop.
        # Here we rely on refitting each outer step (standard in rolling backtests).

        # Fit-step produces sigma2 aligned to data passed into fit()
        # In forecast(), we build the design matrix for all rows and output exp(Xb).
        rv = data["rv_idio"]
        # We need sigma2 for this data slice, so re-fit a GARCH filter quickly:
        r = data["ret_idio"]
        garch_res = arch_model(
            r, mean="zero", vol="GARCH", p=self.p, q=self.q, dist=self.dist, rescale=True
        ).fit(disp="off")

        sigma2 = (garch_res.conditional_volatility ** 2)
        log_sigma2 = np.log(sigma2)

        Xf = pd.DataFrame({
            "log_sigma2": log_sigma2,
            "log_rv_lag1": np.log(rv.shift(1)),
        })
        for col in self.x_cols:
            Xf[col] = data[col].shift(1)

        Xf = sm.add_constant(Xf)
        return np.exp(Xf @ self._ols_params)
