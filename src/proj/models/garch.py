from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

from arch import arch_model


ArrayLike = Union[np.ndarray, list]


class GARCHRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-style wrapper around arch.arch_model for volatility forecasting.

    Notes
    -----
    - `fit(y, X, returns=...)` uses `y` only for its length (to align the returns slice),
      which matches your current workflow (target might be RV/logRV, but GARCH is fit on returns).
    - By default, `predict()` returns *conditional variance* forecasts.
    """

    def __init__(
        self,
        *,
        vol: str = "GARCH",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        power: float = 2.0,
        dist: str = "t",
        mean: Optional[str] = None,          # if None, decided in fit()
        mean_lags: int = 1,                  # mapped to arch_model(..., lags=mean_lags)
        rescale: bool = True,
        horizon: int = 1,                    # default forecast horizon for predict()
        output: Literal["variance", "volatility"] = "variance",
        fit_options: Optional[dict] = None,   # options passed to result.fit(...)
        last_observation_only: bool = True,   # return last available forecast by default
    ):
        self.vol = vol
        self.p = p
        self.o = o
        self.q = q
        self.power = power
        self.dist = dist
        self.mean = mean
        self.mean_lags = mean_lags
        self.rescale = rescale
        self.horizon = horizon
        self.output = output
        self.fit_options = fit_options
        self.last_observation_only = last_observation_only

    # ---------- helpers ----------
    @staticmethod
    def _to_2d(X: Optional[ArrayLike]) -> Optional[np.ndarray]:
        if X is None:
            return None
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D array-like.")
        return X

    @staticmethod
    def _to_1d(y: ArrayLike) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim != 1:
            y = np.asarray(y).reshape(-1)
        return y

    def _build_x_3d(self, X_future: np.ndarray, horizon: int) -> np.ndarray:
        """
        arch expects x with shape (k_exog, horizon, n_scenarios).
        For deterministic 1 scenario: (k, h, 1).
        """
        if X_future.shape[0] != horizon:
            raise ValueError(
                f"X_test must have {horizon} rows for horizon={horizon}, "
                f"got {X_future.shape[0]}."
            )
        k = X_future.shape[1]
        x_3d = np.transpose(X_future[:, :, None], (1, 0, 2))  # (h,k,1)->(k,h,1)
        return x_3d.astype(float, copy=False)

    # ---------- sklearn API ----------
    def fit(self, y: ArrayLike, X: Optional[ArrayLike] = None, *, returns: ArrayLike):
        """
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target series (only length is used to align returns).
        X : array-like, shape (n_samples, n_features), optional
            Exogenous regressors for mean equation (ARX).
        returns : array-like, shape (n_samples_total,)
            Full return series aligned with y. We fit on returns[:len(y)].

        Returns
        -------
        self
        """
        y = self._to_1d(y)
        X = self._to_2d(X)

        returns = self._to_1d(returns)
        n = len(y)
        if len(returns) < n:
            raise ValueError(f"returns length ({len(returns)}) < y length ({n}).")

        r_train = returns[:n]

        mean = self.mean
        if mean is None:
            mean = "ARX" if X is not None else "AR"

        # Build and fit model
        self.model_ = arch_model(
            r_train,
            mean=mean,
            lags=self.mean_lags,
            x=X,
            vol=self.vol,
            p=self.p,
            o=self.o,
            q=self.q,
            power=self.power,
            dist=self.dist,
            rescale=self.rescale,
        )

        fit_opts = {"disp": "off"}
        if self.fit_options:
            fit_opts.update(self.fit_options)

        self.result_ = self.model_.fit(**fit_opts)
        self.n_train_ = n
        self.mean_ = mean
        self.k_exog_ = 0 if X is None else X.shape[1]
        return self

    def predict(self, X: Optional[ArrayLike] = None, *, horizon: Optional[int] = None) -> np.ndarray:
        """
        Forecast conditional variance (or volatility) for `horizon` steps ahead.

        If X is provided (ARX), then X must have shape (horizon, k_exog)
        for multi-step forecasts, or (k_exog,) / (1, k_exog) for horizon=1.

        Returns
        -------
        np.ndarray
            Shape (horizon,) if horizon>1 else shape (1,)
        """
        if not hasattr(self, "result_"):
            raise RuntimeError("Call fit() before predict().")

        h = int(self.horizon if horizon is None else horizon)
        if h < 1:
            raise ValueError("horizon must be >= 1.")

        if self.k_exog_ > 0:
            Xf = self._to_2d(X)
            if Xf is None:
                raise ValueError("Model was fit with exogenous regressors, but X_test is None.")

            # allow providing a single row for h=1
            if h == 1 and Xf.shape[0] != 1:
                # if user passed a vector (k,), _to_2d makes it (n,1) so this is a real mismatch
                raise ValueError(f"For horizon=1, X_test must have 1 row; got {Xf.shape[0]}.")

            # require matching number of columns
            if Xf.shape[1] != self.k_exog_:
                raise ValueError(f"X_test has {Xf.shape[1]} features, expected {self.k_exog_}.")

            if h == 1 and Xf.shape[0] == 1:
                x_3d = self._build_x_3d(Xf, horizon=1)
            else:
                x_3d = self._build_x_3d(Xf, horizon=h)

            f = self.result_.forecast(horizon=h, x=x_3d)
        else:
            if X is not None:
                # ignore silently? better to be strict
                raise ValueError("Model was fit without exogenous regressors; X_test should be None.")
            f = self.result_.forecast(horizon=h)

        # f.variance is a (t, h) DataFrame; take last available t by default.
        var_path = f.variance.values  # shape (t, h)
        var_h = var_path[-1, :] if self.last_observation_only else var_path[:, :]

        if self.output == "volatility":
            out = np.sqrt(var_h)
        else:
            out = var_h

        return np.asarray(out, dtype=float).reshape(-1)

    def score(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Default sklearn score: higher is better.
        We return negative MSE (so lower error -> higher score).
        """
        y_true = self._to_1d(y_true)
        y_pred = self._to_1d(y_pred)
        return -mean_squared_error(y_true, y_pred)

    def summary(self) -> str:
        if not hasattr(self, "result_"):
            raise RuntimeError("Call fit() before summary().")
        return str(self.result_.summary())
