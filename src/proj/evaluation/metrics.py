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

def qlike_arr(realized, forecast, eps=1e-12):
    r = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)

    if r.shape != f.shape:
        raise ValueError(f"Shape mismatch: realized {r.shape}, forecast {f.shape}")

    r = r + eps
    f = f + eps

    ratio = r / f
    q = ratio - np.log(ratio) - 1

    return q



def qlike(realized, forecast, eps=1e-12):
    r = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)

    if r.shape != f.shape:
        raise ValueError(f"Shape mismatch: realized {r.shape}, forecast {f.shape}")

    r = r + eps
    f = f + eps

    ratio = r / f
    q = ratio - np.log(ratio) - 1

    return float(np.mean(q))
