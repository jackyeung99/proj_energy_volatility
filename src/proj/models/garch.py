
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from arch import arch_model

class GARCHRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, returns, **model_params):
        """
        returns: full return series aligned with y_target (same length)

        model_params: arbitrary arch_model parameters.
            Common ones:
                p, q, o, power
                vol: 'GARCH', 'EGARCH', ...
                dist: 'normal', 't', 'skewt', ...
                mean: 'AR', 'ARX', 'Constant', ...
                mean_lags: lags in the mean equation (we map to arch_model's `lags`)
                rescale: True/False
        """
        self.returns = np.asarray(returns)

        # Defaults; user can override any of these
        defaults = {
            "vol": "GARCH",
            "p": 1,
            "q": 1,
            "dist": "t",
            "mean": None,       # will be decided in fit() if not set
            "mean_lags": 1,
            "rescale": True,
        }

        # Merge defaults with user-provided parameters
        self.params = {**defaults, **model_params}

        self._res = None
        self._train_len = None

    def fit(self, y_train, X_train=None):
        """
        y_train: training slice of the TARGET (e.g. RV) — only used for length
        X_train: exogenous regressors for the mean equation, already aligned
        """
        n_train = len(y_train)
        self._train_len = n_train

        r_train = self.returns[:n_train]

        # Copy params so we can tweak without mutating self.params
        params = self.params.copy()

        # Extract mean_lags and map to arch_model's `lags`
        mean_lags = params.pop("mean_lags", 1)

        # Decide mean if user didn't specify explicitly
        if "mean" not in params or params["mean"] is None:
            if X_train is not None:
                params["mean"] = "ARX"
            else:
                params["mean"] = "AR"


    

        if X_train is not None:
            X_train = np.asarray(X_train)
            am = arch_model(
                r_train,
                lags=mean_lags,
                x=X_train,   # exogenous in mean
                **params     # e.g. vol, p, q, dist, rescale, etc.
            )
        else:
            am = arch_model(
                r_train,
                lags=mean_lags,
                **params
            )

        self._res = am.fit(disp="off")
        return self._res

    def summary(self,):

        return self.summary()

    def predict(self, X_test=None):
        if self._res is None:
            raise RuntimeError("Call fit() before predict().")

        if X_test is not None:
            X_test = np.asarray(X_test)
            # single 1-step forecast
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            if X_test.shape[0] != 1:
                raise ValueError("With exogenous regressors, only 1-step forecasts are supported.")
            n_exog = X_test.shape[1]

            x_3d = np.empty((n_exog, 1, 1), dtype=float)
            for j in range(n_exog):
                x_3d[j, 0, 0] = X_test[0, j]

            f = self._res.forecast(horizon=1, x=x_3d)
        else:
            f = self._res.forecast(horizon=1)


        # last row, 1-step ahead variance
        return float(f.variance.values[-1, 0])



