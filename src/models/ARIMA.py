# from typing import tuple, int
from sklearn.linear_model import LinearRegression
import numpy as np
import math
from scipy.linalg import solve_discrete_lyapunov
from numpy.linalg import inv, det
from scipy.optimize import minimize
import matplotlib.pyplot as plt



class ARIMA_model():
    def __init__(self, time_series: np.array, order: tuple[int, int, int]):
        self.p, self.l, self.q = order
        self.r = max(self.p, self.q + 1)
        self.original_time_series = time_series
        self.y = self.differencing(self.l, time_series)
    


    # ============================== Orchestration ==============================
    def fit(self, method="L-BFGS-B"):
        # ---- initialize params (AR from OLS, MA small, sigma^2 from var of residual) ----
        alpha0, beta0 = self.init_ARMA_parameters()
        sigma20 = float(np.var(self.y - np.mean(self.y))) if len(self.y) > 1 else 1.0
        theta0 = self.pack_params(alpha0, beta0, sigma20)

        # ---- optimize negative log-likelihood ----
        res = minimize(self.neg_loglik, theta0, method=method, options={"maxiter": 500})

        # ---- store results ----
        self.theta_hat = self.unpack_params(res.x)
        self.mle_result = res
        return self.theta_hat, res


    # ============================== Forecasts  ==============================
    def forecast_change(self, steps=5):
        """
        Forecast future values (in-sample differenced scale) for given number of steps.
        Returns forecasted means and confidence intervals.
        """
        if not hasattr(self, "theta_hat"):
            raise ValueError("You must fit the model before forecasting.")
        
        alpha, beta, sigma2 = self.theta_hat
        T, A, R, Q, H = self.build_state_space_model(alpha, beta, sigma2)

        # Re-run Kalman filter through full data to get last state
        w, Omega = self.initialize_kalman_filter(T, R, Q, alpha)
        for y_t in self.y:
            y_t_mat = np.array([[float(y_t)]])
            w_pred, y_pred, Omega_pred, S = self.prediction(T, A, R, Q, H, w, Omega)
            v = y_t_mat - y_pred
            K = self.kalman_gain(Omega_pred, A, S)
            w, Omega = self.kalman_update(w_pred, Omega_pred, K, A, v)

        # Forecasting loop
        forecasts = []
        variances = []
        for h in range(steps):
            w = T @ w
            Omega = T @ Omega @ T.T + R @ Q @ R.T
            y_fore = A @ w
            y_var = A @ Omega @ A.T
            forecasts.append(float(y_fore))
            variances.append(float(y_var))

        forecasts = np.array(forecasts)
        se = np.sqrt(np.array(variances))

        z = 1.96  # for 95% CI
        lower = forecasts - z * se
        upper = forecasts + z * se

        return forecasts, lower, upper
    

    def invert_forecast_diff(self, diffs, d=None):
        if d is None:
            d = self.l
            
        levels = np.asarray(self.original_time_series, dtype=float)
        cur = diffs.copy()  
        for k in range(d, 0, -1):
            if k == 1:
                base_k = float(levels[-1])                # y_T
            else:
                base_k = float(np.diff(levels, n=k-1)[-1])  # Δ^{k-1} y_T
            cur = base_k + np.cumsum(cur)

        return cur




    
    def forecast(self, steps=5):
        forecast, lower, upper = self.forecast_change(steps)

        forecast = self.invert_forecast_diff(forecast)
        lower = self.invert_forecast_diff(lower)
        upper = self.invert_forecast_diff(upper)

        return forecast, lower, upper



    def fitted_values(self):
        """
        Run Kalman filter one more time to produce in-sample fitted values.
        Returns arrays of y_pred (one-step-ahead) and y_true.
        """
        alpha, beta, sigma2 = self.theta_hat
        T, A, R, Q, H = self.build_state_space_model(alpha, beta, sigma2)
        w, Omega = self.initialize_kalman_filter(T, R, Q, alpha)

        y_true, y_fitted = [], []

        for y_t in self.y:
            y_t_mat = np.array([[float(y_t)]])
            w_pred, y_pred, Omega_pred, S = self.prediction(T, A, R, Q, H, w, Omega)

            y_true.append(float(y_t))
            y_fitted.append(float(y_pred))  

            # Update step
            v = y_t_mat - y_pred
            K = self.kalman_gain(Omega_pred, A, S)
            w, Omega = self.kalman_update(w_pred, Omega_pred, K, A, v)

            

        return np.array(y_true), np.array(y_fitted)
    #============================== Step 1: Initialization ==============================
    def ols_fit(self, y, lag):
        p = lag
        if p == 0:
            return np.array([])
        Y = y[p:]
        # lag matrix
        X = np.column_stack([y[p - i - 1: -i - 1] for i in range(p)])
        XtX = X.T @ X
        XtY = X.T @ Y
        beta_hat = np.linalg.solve(XtX, XtY)
        return beta_hat

    def init_ARMA_parameters(self):
        alpha_hat = self.ols_fit(self.y, self.p)
        beta_hat = np.array([.1 for _ in range(self.q)])
        return alpha_hat, beta_hat
    



    #============================== Parameter packing ==============================
    def pack_params(self, alpha, beta, sigma2):
        return np.concatenate([np.asarray(alpha, float),
                               np.asarray(beta, float),
                               np.array([np.log(sigma2)], float)])

    def unpack_params(self, theta_vec):
        p, q = self.p, self.q
        alpha = theta_vec[:p] if p > 0 else np.array([])
        beta  = theta_vec[p:p+q] if q > 0 else np.array([])
        sigma2 = float(np.exp(theta_vec[-1]))
        return alpha, beta, sigma2


    #============================== State-space builder ==============================
    def build_state_space_model(self, alpha, beta, sigma_eps2=1.0):
        """
        measurement: y_t = A w_t + u_t,   u_t ~ N(0, H)
        transition:  w_t = T w_{t-1} + R e_t,  e_t ~ N(0, Q)

        We inject the innovation into the first state via R, and set H=0 (since
        the measurement noise is handled via the state/innovation).
        """
        # T
        T = np.zeros((self.r, self.r))
        if self.p > 0:
            T[0, :self.p] = np.asarray(alpha, float)
        if self.r > 1:
            T[1:, :-1] = np.eye(self.r - 1)

        # A (loads states to observed y_t). Include MA lags if q>0 in the state order.
        A = np.zeros((1, self.r))
        A[0, 0] = 1.0
        if self.q > 0:
            A[0, 1:1 + self.q] = np.asarray(beta, float)

        # R maps innovation into the state vector (first element)
        R = np.zeros((self.r, 1))
        R[0, 0] = 1.0

        # Q is variance of the innovation e_t
        Q = np.array([[float(sigma_eps2)]])

        # H is measurement noise variance; here set to 0 since innovation is in the state
        H = np.array([[0.0]])

        return T, A, R, Q, H

    def initialize_kalman_filter(self, T, R, Q, alpha_hat):
        w0 = np.zeros((self.r, 1))
        # stationary initialization via Lyapunov where possible
        # if self.poly_roots_outside_unit(alpha_hat, kind="AR"):
        #     Omega0 = solve_discrete_lyapunov(T, R @ Q @ R.T)
        # else:
        Omega0 = np.eye(self.r) * 1e6
        return w0, Omega0


    #============================== Step 2: Prediction ==============================
    def prediction(self, T, A, R, Q, H, w, Omega):
        # state pred
        w_pred = T @ w
        # covariance pred: Omega_{t|t-1} = T Omega T' + R Q R'
        Omega_pred = T @ Omega @ T.T + R @ Q @ R.T
        # y pred
        y_pred = A @ w_pred
        # innovation variance: S = A Omega_pred A' + H
        S = A @ Omega_pred @ A.T + H
        return w_pred, y_pred, Omega_pred, S

    #============================== Step 3: Likelihood (per-step) ==============================
    def conditional_log_likelihood(self, S, v):
        k = S.shape[0]
        # stable inverse for 1x1 case
        if k == 1:
            s = float(S[0, 0])
            return -0.5 * (np.log(2*np.pi) + np.log(s) + (float(v)**2) / s)
        # generic multivariate
        return (-0.5 * (k * np.log(2*np.pi) + np.log(det(S)) + (v.T @ inv(S) @ v))).item()
    

    #============================== Step 4: Update/Kalman Filter ==============================
    def kalman_gain(self, Omega_pred, A, S):
        # Ensure symmetry + tiny jitter for stability
        S = 0.5 * (S + S.T) + 1e-8 * np.eye(S.shape[0])

        if S.shape == (1, 1):
            invS = 1.0 / float(S[0, 0])
            return Omega_pred @ A.T * invS 
        
        rhs = A @ Omega_pred               
        X = np.linalg.solve(S, rhs)        
        K = X.T                            
        return K

    def kalman_update(self, w_pred, Omega_pred, K, A, v):
        w_new = w_pred + K @ v
        Omega_new = Omega_pred - K @ A @ Omega_pred
        return w_new, Omega_new

    #============================== Full-sample negative log-likelihood ==============================
    def neg_loglik(self, theta_vec):
        alpha, beta, sigma2 = self.unpack_params(theta_vec)


        # Enforce stationarity/invertibility (hard check with penalty)
        if not self.poly_roots_outside_unit(alpha, kind="AR"):  return 1e12
        if not self.poly_roots_outside_unit(beta, kind="MA"):  return 1e12
        if not np.isfinite(sigma2) or sigma2 <= 1e-12:     return 1e12

        T, A, R, Q, H = self.build_state_space_model(alpha, beta, sigma2)
        w, Omega = self.initialize_kalman_filter(T, R, Q, alpha)

        ll = 0.0
        for t, y_t in enumerate(self.y):
            y_t_mat = np.array([[float(y_t)]])
            w_pred, y_pred, Omega_pred, S = self.prediction(T, A, R, Q, H, w, Omega)

            v = y_t_mat - y_pred
            ll_t = self.conditional_log_likelihood(S, y_t_mat - y_pred)
            if not np.isfinite(ll_t):
                return 1e12
            ll += ll_t

            K = self.kalman_gain(Omega_pred, A, S)
            w, Omega = self.kalman_update(w_pred, Omega_pred, K, A, v)


        return -ll 
    
    #============================== Helpers ==============================

    def poly_roots_outside_unit(self, coeffs, kind="AR"):
        coeffs = np.asarray(coeffs, float)[::-1]
        if coeffs.size == 0:
            return True
        poly = np.r_[(-coeffs if kind=="AR" else coeffs), 1.0]
        roots = np.roots(poly)
        return np.all(np.abs(roots) > 1.0)
    

    def differencing(self, l, time_series):
        self.init_value_diff = []  #
        ts = np.asarray(time_series, float)
        for _ in range(l):
            self.init_value_diff.append(ts[0])   
            ts = np.diff(ts)                     # first difference
        return ts  
    
    def invert_differencing(self, diffs, d=None, mode="observed", levels=None):
        if d is None:
            d = int(self.l)
        diffs = np.asarray(diffs, float)

        if d == 0:
            # No differencing: diffs already are levels
            return diffs.copy()

        if mode == "observed":

            out = diffs.copy()
            # Integrate upward d times, each time seeding with the recorded first value
            for k in range(d):
                base = float(self.init_value_diff[k])
                out = np.r_[base, base + np.cumsum(out)]
            return out

        elif mode == "fitted":
            if levels:
                levels = np.asarray(levels, float)
            else:
                levels = self.original_time_series

            out = diffs.copy()
            # Integrate upward d times, each time seeding with the recorded first value using actual values
            for k in range(d, 0, -1):
                stage_actual = np.diff(levels, n=k-1) if k > 1 else levels
                out = stage_actual[:-1] + out  

            pad = np.full(d, np.nan)
            out= np.r_[pad, out]
            return out

        else:
            raise ValueError("mode must be 'observed' or 'fitted'")




    def detrend(self, time_series): 
       return time_series - time_series.mean()
    


    
    #============================== Metrics ==============================
    def compute_aic(self, log_like, n_params, n_obs):
        """Compute Akaike Information Criterion."""
        return 2 * n_params - 2 * log_like


    def compute_bic(self, log_like, n_params, n_obs):
        """Compute Bayesian Information Criterion."""
        return np.log(n_obs) * n_params - 2 * log_like


    def model_summary(self):
        """
        Summarize fitted model:
        - AR and MA parameters
        - Log-likelihood
        - AIC, BIC
        """
        if not hasattr(self, "mle_result"):
            raise ValueError("You need to call fit() before model_summary().")

        # Unpack parameters
        alpha, beta, sigma2 = self.theta_hat
        log_like = -self.mle_result.fun   
        n_params = len(alpha) + len(beta) + 1
        n_obs = len(self.y)

        aic = self.compute_aic(log_like, n_params, n_obs)
        bic = self.compute_bic(log_like, n_params, n_obs)

        print("=========================================")
        print(f" ARIMA({self.p},{self.i},{self.q}) Model Summary")
        print("=========================================")
        print(f"Number of observations: {n_obs}")
        print(f"Number of parameters  : {n_params}")
        print("-----------------------------------------")
        if len(alpha) > 0:
            for i, a in enumerate(alpha, 1):
                print(f"AR({i}) coefficient: {a: .4f}")
        if len(beta) > 0:
            for j, b in enumerate(beta, 1):
                print(f"MA({j}) coefficient: {b: .4f}")
        print(f"Variance (sigma^2): {sigma2: .4f}")
        print("-----------------------------------------")
        print(f"Log-likelihood: {log_like: .4f}")
        print(f"AIC: {aic: .4f}")
        print(f"BIC: {bic: .4f}")
        print("=========================================")

        return {
            "log_likelihood": log_like,
            "aic": aic,
            "bic": bic,
            "params": {"ar": alpha, "ma": beta, "sigma2": sigma2},
        }


    def plot_fit(self):
        """
        Plot observed vs fitted series.
        """
        y_true, y_fitted = self.fitted_values()
        y_true = self.invert_differencing(y_true,mode="observed")
        y_fitted = self.invert_differencing(y_fitted, mode="fitted")

        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Observed", linewidth=2)
        plt.plot(y_fitted, label="Fitted (one-step ahead)", linestyle='--', linewidth=2)
        plt.title(f"ARMA({self.p},{self.q}) Fit via Kalman Filter")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
