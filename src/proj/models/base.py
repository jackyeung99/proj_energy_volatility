# models/base.py
from abc import ABC, abstractmethod
import pandas as pd

class VolatilityModel(ABC):
    """
    All models forecast variance.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fit model using training data"""
        pass

    @abstractmethod
    def forecast(self, data: pd.DataFrame) -> pd.Series:
        """
        Produce 1-step-ahead variance forecast aligned with data index
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
