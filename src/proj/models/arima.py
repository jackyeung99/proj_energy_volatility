import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class ARIMA_RV_Model:
    def __init__(self, **model_params):
        """
        Example params:
            order=(p, d, q)
            trend='n' | 'c' | 't' | 'ct'
            seasonal_order=(P, D, Q, s)
            enforce_stationarity=True/False
            ...
        """

        # ---------- non-seasonal order: (p,d,q) ----------
        if "order" in model_params:
            order = model_params["order"]
        else:
            p = model_params.pop("p", 0)
            d = model_params.pop("d", 0)
            q = model_params.pop("q", 0)
            order = (p, d, q)

        order = tuple(order)

        # ---------- seasonal order: (P,D,Q,s) ----------
        if "seasonal_order" in model_params:
            seasonal_order = model_params["seasonal_order"]
        else:
            # check if user passed any seasonal components
            has_seasonal = any(
                k in model_params for k in ("seasonal_P", "seasonal_D", "seasonal_Q", "seasonal_s")
            )
            if has_seasonal:
                P = model_params.pop("seasonal_P", 0)
                D = model_params.pop("seasonal_D", 0)
                Q = model_params.pop("seasonal_Q", 0)
                s = model_params.pop("seasonal_s", 0)
                seasonal_order = (P, D, Q, s)
            else:
                # default: no seasonality
                seasonal_order = (0, 0, 0, 0)

        seasonal_order = tuple(seasonal_order)

        self.params = {
            **model_params,
            "order": order,
            "seasonal_order": seasonal_order,
        }

        self._model = None
        self._res = None

    def fit(self, y, X=None):
        y = np.asarray(y)
        params = self.params.copy()

        # we'll pass exog explicitly
        params.pop("exog", None)

        if X is not None:
            X = np.asarray(X)
            self._model = ARIMA(endog=y, exog=X, **params)
        else:
            self._model = ARIMA(endog=y, **params)

        self._res = self._model.fit()
        return self

    def predict(self, horizon=1, X_test=None):
        if self._res is None:
            raise RuntimeError("Call fit() before predict().")

        if X_test is not None:
            X_test = np.asarray(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            if X_test.shape[0] != horizon:
                raise ValueError(
                    f"X_test has {X_test.shape[0]} rows but horizon={horizon}."
                )
            fcst_res = self._res.get_forecast(steps=horizon, exog=X_test)
        else:
            fcst_res = self._res.get_forecast(steps=horizon)

        return np.asarray(fcst_res.predicted_mean)

    # optional sklearn-style hooks
    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **new_params):
        self.params.update(new_params)
        return self
