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

    



