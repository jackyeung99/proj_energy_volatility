
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from arch import arch_model

class GARCHRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        p=1,
        q=1,
        dist="normal",
        mean="zero",
        vol="Garch",
        rescale=True,
    ):
        # hyperparams must be stored with same names
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.vol = vol
        self.rescale = rescale

        # will be set in fit
        self._model_ = None
        self._res_ = None

    def fit(self, X, y):
        """
        X is ignored in this simplest version (univariate GARCH).
        y should be a 1D array-like of returns.
        """
        y = np.asarray(y, dtype=float).ravel()

        self._model_ = arch_model(
            y,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=self.rescale,
        )
        self._res_ = self._model_.fit(disp="off")

        # sklearn wants fit() to return self
        return self

    def predict(self, X=None):
        """
        Return in-sample conditional volatility for the training sample.

        If you want out-of-sample forecasts that depend on horizon,
        you could add a horizon argument or a separate method.
        """
        if self._res_ is None:
            raise RuntimeError("You must call fit before predict.")

        # conditional_volatility is aligned with y used in fit
        cond_vol = self._res_.conditional_volatility
        return np.asarray(cond_vol)

    def forecast(self, horizon=1):
        """
        Convenience method for out-of-sample forecasts, not used by sklearn directly.
        """
        if self._res_ is None:
            raise RuntimeError("You must call fit before forecast.")

        f = self._res_.forecast(horizon=horizon)
        # variance forecast for last point, horizon steps ahead
        return np.sqrt(f.variance.values[-1])