# src/proj/pipelines/regime.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np 

from proj.models.regime_switching import *



from proj.utils.dates import utc_run_id

import logging
logger = logging.getLogger("proj.regime")


def regime(storage, global_cfg: dict, step_cfg: dict) -> pd.DataFrame:
    """
    Fit Markov regime model on realized volatility (gold) and write a daily regime table.

    Output (one row per ET trading date):
      - asof_date_et (join key)
      - p_state_i (filtered probs)
      - p_state_i_next (one-step-ahead probs)
      - state_label / state_label_next (Low/Medium/High ordered by mean)
      - confidence / confidence_next
      - run_id (latest run identifier)
    """
    run_id = utc_run_id()

    gold_path, out_path, realized_col, eps, window_bd, markov_cfg = parse_regime_cfg(step_cfg)

    gold = storage.read_parquet(gold_path)

    y = np.log(gold[realized_col] + eps)

    out = compute_markov_regimes_daily(
        y=y,
        markov_cfg=markov_cfg,
        run_id=run_id,
    )


    storage.write_parquet(out, out_path)
  
    logger.info("Wrote regimes table with %d rows to %s", len(out), out_path)

    return out


def parse_regime_cfg(step_cfg: dict) -> Tuple[str, str, str, float, int | None, Dict[str, Any]]:
    """
    Returns:
      gold_path, out_path, realized_col, eps, window_bd, markov_cfg
    """
    files_cfg = step_cfg.get("files", {}) or {}
    gold_path = files_cfg.get("data_dir")
    out_path = files_cfg.get("output_dir")

    if not gold_path:
        raise ValueError("regime cfg missing files.rv_data_dir")
    if not out_path:
        raise ValueError("regime cfg missing files.output_dir")

    reg_cfg = step_cfg.get("regime", {}) or {}

    observable = reg_cfg.get("observable", {}) or {}
    realized_col = str(observable.get("column", "rv_xle"))
    eps = float(observable.get("eps", 1e-12))

    window_bd = reg_cfg.get("window_bd")
    window_bd = int(window_bd) if window_bd is not None else None

    markov_cfg = reg_cfg.get("markov", {}) or {}
    # set defaults here so call site stays simple
    markov_cfg = {
        "k_states": int(markov_cfg.get("k_states", 3)),
        "switching_variance": bool(markov_cfg.get("switching_variance", True)),
        "trend": str(markov_cfg.get("trend", "c")),
        "em_iter": int(markov_cfg.get("em_iter", 10)),
        "maxiter": int(markov_cfg.get("maxiter", 200)),
    }

    return gold_path, out_path, realized_col, eps, window_bd, markov_cfg




def compute_markov_regimes_daily(y: pd.Series, markov_cfg: Dict[str, Any], run_id: str) -> pd.DataFrame:
    fit = fit_markov_regime_model(
        y=y,
        k_states=int(markov_cfg["k_states"]),
        switching_variance=bool(markov_cfg["switching_variance"]),
        trend=str(markov_cfg["trend"]),
        em_iter=int(markov_cfg["em_iter"]),
        maxiter=int(markov_cfg["maxiter"]),
    )
    return build_regime_table(fit, run_id=run_id)