
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import arch

class GARCHRegressor:
    def __init__(self, returns, p=1, q=1, dist="t"):
        """
        returns: full return series aligned with y_target (same length)
        """
        self.returns = np.asarray(returns)
        self.p = p
        self.q = q
        self.dist = dist
        self._res = None
        self._train_len = None

    def fit(self, y_train, X_train=None):
        """
        y_train: training slice of the TARGET (RV) — we only use its length
        to know where to cut the returns.
        """
        n_train = len(y_train)
        self._train_len = n_train

        r_train = self.returns[:n_train]

        am = arch.univariate.ARX(r_train, lags=0, rescale=True)
        am.volatility = arch.univariate.GARCH(p=self.p, q=self.q)

        if self.dist == "normal":
            am.distribution = arch.univariate.Normal()
        elif self.dist == "t":
            am.distribution = arch.univariate.StudentsT()
        else:
            raise ValueError("Unknown dist")

        self._res = am.fit(disp="off")

    def predict(self, horizon, X_future=None):
        # Forecast conditional variance for next `horizon` days
        fcast = self._res.forecast(horizon=horizon, start=self._train_len - 1)
        var_forecast = fcast.variance.iloc[-1].values  # shape (horizon,)
        # convert to realized VOL (not variance)
        rv_pred = np.sqrt(var_forecast)
        return rv_pred