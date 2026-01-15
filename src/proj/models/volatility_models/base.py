# models/base.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class VolatilityModel(ABC):
    """
    All models forecast variance.
    """

    fitted_variance_: Optional[pd.Series] = None  # set during fit()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fit model using training data and store in-sample fitted variance."""
        ...

    @abstractmethod
    def forecast(self, data: pd.DataFrame) -> pd.Series:
        """Produce 1-step-ahead variance forecast aligned with data index."""
        ...

    def fitted(self) -> pd.Series:
        """
        Return in-sample fitted variance from the most recent fit().
        """
        if self.fitted_variance_ is None:
            raise RuntimeError("No fitted variance available. Call fit() first.")
        return self.fitted_variance_
