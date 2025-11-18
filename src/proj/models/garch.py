
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from arch import arch_model

class GARCHRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, returns, p=1, q=1, dist="t", mean_lags=1):
        """
        returns: full return series aligned with y_target (same length)
        p, q: GARCH(p, q)
        dist: 'normal' or 't'
        mean_lags: lags in the ARX mean equation
        """
        self.returns = np.asarray(returns)
        self.p = p
        self.q = q
        self.dist = dist
        self.mean_lags = mean_lags

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

        if X_train is not None:
            X_train = np.asarray(X_train)

            am = arch_model(
                r_train,
                mean="ARX",
                lags=self.mean_lags,
                x=X_train,          # exogenous regressors in mean
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,     # 't' or 'normal'
                rescale=True
            )
        else:
            # no exogenous variables, just AR in the mean
            am = arch_model(
                r_train,
                mean="AR",
                lags=self.mean_lags,
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=True
            )

        self._res = am.fit(disp="off")
        return self

    def predict(self, horizon=1, X_test=None):
        """
        Forecast variance over `horizon` steps ahead.
        If the model was fit with exogenous variables, X_test must be provided
        with shape (horizon, n_exog).
        """
        if self._res is None:
            raise RuntimeError("Call fit() before predict().")
        
        if X_test is not None:       
            # arch_model expects x with same number of columns as in-sample X
            X_test = np.asarray(X_test)


            # Allow 1D (single regressor, horizon=1) or 2D
            if X_test.ndim == 1:
                # treat as horizon=1, n_exog = len(X_test)
                X_test = X_test.reshape(1, -1)

            # X_test is now (horizon, n_exog)
            if X_test.shape[0] != horizon:
                raise ValueError(
                    f"X_test has {X_test.shape[0]} rows but horizon={horizon}. "
                    "Expected X_test.shape[0] == horizon."
                )

            horizon_ = horizon
            n_exog = X_test.shape[1]

            # Build 3D array: (n_exog, nforecast, horizon)
            # Here nforecast = 1 (we forecast from one origin, at the end of sample)
            x_3d = np.empty((n_exog, 1, horizon_), dtype=float)
            # For regressor j, we need shape (1, horizon)
            for j in range(n_exog):
                x_3d[j, 0, :] = X_test[:, j]

            f = self._res.forecast(horizon=horizon_, x=x_3d)
        else:
            f = self._res.forecast(horizon=horizon)

        # Return the variance forecasts for the horizon
        # (last row corresponds to the out-of-sample forecasts)
        return f.variance.values[-1, :]