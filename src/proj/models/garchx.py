import numpy as np
from arch.univariate import GARCH


class GARCHX(GARCH):
    """
    GARCH-X volatility model:

        sigma_t^power = omega
                        + sum alpha_i * |eps_{t-i}|^power
                        + sum beta_j * sigma_{t-j}^power
                        + gamma' x_t

    where x_t are exogenous regressors for the volatility equation.
    """

    def __init__(self, p=1, q=1, x=None, power=2.0):
        super().__init__(p=p, o=0, q=q, power=power)

        if x is None:
            raise ValueError("You must supply exogenous regressors x for GARCH-X.")

        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]  # convert to 2D (T, 1)

        self.x = x
        self.kx = x.shape[1]  # number of exogenous regressors

        # Original GARCH has params: omega + p alphas + q betas
        # GARCH-X adds kx gamma parameters
        self._num_params = 1 + p + q + self.kx
        self._name = "GARCHX"

    # -------------------------------------------------------------
    def parameter_names(self):
        base = super().parameter_names()
        gammas = [f"gamma[{i+1}]" for i in range(self.kx)]
        return base + gammas

    # -------------------------------------------------------------
    def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):
        power = self.power
        p, q = self.p, self.q
        x = self.x

        nobs = len(resids)
        abs_eps_pow = np.abs(resids)**power

        omega = parameters[0]
        alpha = parameters[1 : 1+p]
        beta  = parameters[1+p : 1+p+q]
        gamma = parameters[1+p+q : 1+p+q+self.kx]

        # variance in power domain
        s = np.zeros(nobs)
        s[0] = float(backcast)

        for t in range(1, nobs):
            arch_part = sum(alpha[i] * abs_eps_pow[t-1-i]
                            for i in range(p) if t-1-i >= 0)
            garch_part = sum(beta[j] * s[t-1-j]
                             for j in range(q) if t-1-j >= 0)
            x_part = float(np.dot(gamma, x[t]))

            st = omega + arch_part + garch_part + x_part
            st = np.clip(st, var_bounds[t, 0], var_bounds[t, 1])

            s[t] = st

        inv_power = 2.0 / power
        sigma2[:] = s**inv_power
        return sigma2
