from xgboost import XGBRegressor


class XGBRVModel:
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model_ = XGBRegressor(**xgb_params)

    def fit(self, y_train, X_train=None):
        if X_train is None:
            raise ValueError("XGBRVModel needs X features.")
        self.model_.fit(X_train, y_train)

    def predict(self, horizon, X_test=None):
        # direct multi-step: X_future already encoded for each horizon step
        if X_test is None:
            raise ValueError("Need X_test for prediction.")
        return self.model_.predict(X_test)