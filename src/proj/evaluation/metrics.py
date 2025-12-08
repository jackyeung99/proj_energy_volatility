import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import het_arch
from sklearn.metrics import mean_squared_error

def test_differencing(series):
    pass

def test_cointegration(series1, series2):
    pass


def test_for_garch(series):
    pass



def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return rmse


def log_loss(var_true, var_pred):
    ratio = var_true / var_pred
    return np.mean()


def qlike(var_true, var_pred):
    """
    QLIKE loss for variance forecasts.
    Inputs MUST be variances, not volatilities.
    """
    var_true = np.asarray(var_true, dtype=float)
    var_pred = np.asarray(var_pred, dtype=float)

    if np.any(var_true < 0):
        raise ValueError("var_true must be nonnegative.")
    if np.any(var_pred <= 0):
        raise ValueError("var_pred must be strictly positive.")

    ratio = var_true / var_pred
    return np.mean(ratio - np.log(ratio) - 1)


