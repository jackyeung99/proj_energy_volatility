import numpy as np
import statsmodels.api as sm

class ARIMA_RV_Model:
    def __init__(self, order=(1,0,0)):
        """
        order: (p, d, q) ARIMA order
        """
        self.order = order
        self._model = None     # statsmodels model
        self._res = None       # fitted result

    def fit(self, y_train, X_train=None):
        """
        y_train : 1D array
        X_train : 2D array (optional)
        """
        y_train = np.asarray(y_train)

        if X_train is None:
            self._model = sm.tsa.ARIMA(endog=y_train, order=self.order)
        else:
            self._model = sm.tsa.ARIMA(endog=y_train, exog=X_train, order=self.order)

        self._res = self._model.fit()

    def predict(self, horizon, X_future=None):
        """
        horizon : int, number of steps ahead
        X_future : exogenous regressors for forecast period
        """
        if X_future is None:
            forecast = self._res.forecast(steps=horizon)
        else:
            forecast = self._res.forecast(steps=horizon, exog=X_future)

        return np.asarray(forecast)
