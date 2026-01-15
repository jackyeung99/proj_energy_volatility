# models/garch_x_proxy.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from .base import VolatilityModel

class GARCHProxyX(VolatilityModel):
    """
    Two-step model:
      1) Fit GARCH on returns -> sigma2_t (filtered)
      2) Fit log(RV_t) ~ log(sigma2_t) + log(RV_{t-1}) + X_{t-1}
    Outputs variance forecasts (in RV units).
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

    def _build_reg_df(self, data: pd.DataFrame, log_sigma2: pd.Series) -> pd.DataFrame:
        """
        Build regression dataframe aligned to data index.
        """
        rv = data["rv"]
        df = pd.DataFrame(index=data.index)
        df["y"] = np.log(rv)
        df["log_sigma2"] = log_sigma2
        df["log_rv_lag1"] = np.log(rv.shift(1))

        for col in self.x_cols:
            df[col] = data[col].shift(1)

        return df

    def fit(self, data: pd.DataFrame):
        
        # ---- Step 1: fit GARCH on returns (training window)
        r = data["ret"]
        self._garch_res = arch_model(
            r, mean="zero", vol="GARCH",
            p=self.p, q=self.q, dist=self.dist, rescale=True
        ).fit(disp="off")

        sigma2 = (self._garch_res.conditional_volatility ** 2)
        sigma2 = pd.Series(sigma2, index=data.index, name="sigma2")
        log_sigma2 = np.log(sigma2.replace(0, np.nan))

        # ---- Step 2: OLS on log RV with lagged predictors + X
        reg = self._build_reg_df(data, log_sigma2).dropna()
        X = sm.add_constant(reg.drop(columns=["y"]))
        y = reg["y"]

        self._ols_res = sm.OLS(y, X).fit()
        self._ols_params = self._ols_res.params

        # ---- In-sample fitted variance series (aligned to training index)
        fitted_log = self._ols_res.fittedvalues
        fitted_var = np.exp(fitted_log)
        self.fitted_variance_ = fitted_var.reindex(data.index)

        return self

    def forecast(self, data: pd.DataFrame) -> pd.Series:
        """
        Return variance forecasts aligned to `data` index.
        IMPORTANT: This method assumes you're refitting the whole model in the outer loop
        (rolling/expanding backtest). If you are NOT refitting, you must supply
        sigma2 computed using the fitted GARCH parameters (requires a custom filter).
        """
        if self._ols_params is None:
            raise RuntimeError("Call fit() before forecast().")

        # Re-fit GARCH on the provided slice to get sigma2 aligned to this slice.
        # (Consistent with your rolling refit setup.)
        r = data["ret"]
        garch_res = arch_model(
            r, mean="zero", vol="GARCH",
            p=self.p, q=self.q, dist=self.dist, rescale=True
        ).fit(disp="off")

        sigma2 = (garch_res.conditional_volatility ** 2)
        sigma2 = pd.Series(sigma2, index=data.index, name="sigma2")
        log_sigma2 = np.log(sigma2.replace(0, np.nan))

        regf = self._build_reg_df(data, log_sigma2).dropna()
        Xf = sm.add_constant(regf.drop(columns=["y"]), has_constant="add")

        # exp(Xb) gives variance forecast in RV units
        yhat = np.exp(Xf @ self._ols_params)
        return yhat.iloc[-1:]
