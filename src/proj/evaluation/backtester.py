# evaluation/backtester.py
import pandas as pd
from proj.evaluation.engine import walkforward_points
from proj.evaluation.metrics import qlike

class Backtester:
    """
    Owns:
    - time splits
    - fit / forecast protocol
    - scoring

    Supports initial training window specified by:
      - absolute count (initial_train)
      - percentage of sample (train_pct)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metric=qlike, 
        target_col: str = "rv",
        initial_train: int | None = None,
        train_pct: float | None = None,
        step: int = 1,
    ):
        self.data = data
        self.metric = metric
        self.target_col = target_col
        self.step = step

        n = len(data)

        # ---- handle train window specification
        if (initial_train is None) == (train_pct is None):
            raise ValueError(
                "Specify exactly one of initial_train or train_pct."
            )

        if train_pct is not None:
            if not (0 < train_pct < 1):
                raise ValueError("train_pct must be in (0, 1).")
            self.initial_train = int(round(train_pct * n))
        else:
            if initial_train <= 0 or initial_train >= n:
                raise ValueError("initial_train must be in [1, n-1].")
            self.initial_train = int(initial_train)

        self.results_ = None

    def run(self, model, store: bool = True) -> pd.DataFrame:
        """
        Full expanding-window walk-forward backtest.
        """
        rows = []

        for train_end, test_idx in walkforward_points(
            len(self.data), self.initial_train, self.step
        ):
            train = self.data.iloc[:train_end]
            test  = self.data.iloc[[test_idx]]

            model.fit(train)
            f = float(model.forecast(pd.concat([train, test])).iloc[-1])
            y = float(test[self.target_col].iloc[0])

            rows.append({
                "date": self.data.index[test_idx],
                "model": model.name,
                "forecast": f,
                "truth": y,
            })

        res = pd.DataFrame(rows)

        if store:
            self.results_ = res

        return res

    def score_model(self, model) -> float:
        """
        Convenience method for tuners / feature selection.
        Returns mean score.
        """
        res = self.run(model, store=False)
        return self.metric(res["truth"], res["forecast"])
