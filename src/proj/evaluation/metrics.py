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

    
def qlike(y_true, y_pred):
    """
    QLIKE loss for volatility forecasting.

    y_true : array-like 
        Realized volatility OR realized variance.
        If volatility, it will automatically be squared.
    y_pred : array-like
        Forecast volatility OR forecast variance.
        If volatility, it will automatically be squared.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_var = y_true**2
    pred_var = y_pred**2

    ratio = true_var / pred_var
    return np.mean(ratio - np.log(ratio) - 1)


