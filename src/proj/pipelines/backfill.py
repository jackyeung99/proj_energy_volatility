from __future__ import annotations

import os
import argparse
from pathlib import Path

import pandas as pd
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from proj.utils.dates import  *

from proj.utils.config import load_config
from proj.utils.logging import setup_logging, log_step
from proj.data.storage import make_storage


from proj.pipelines.steps.scoring import score_predictions

from proj.pipelines.steps._predict_utils import (
    parse_predict_cfg,
    apply_train_window,
    load_and_standardize_modeling_table,
    resolve_prediction_times,
    get_enabled_model_specs,
    forecast_models,
    merge_and_persist_predictions,
)


from typing import Optional

# ----------------------------
# HARDCODED CONFIG PATHS
# ----------------------------
RUN_ALL_CFG = Path("configs/run_all.yaml")
PRED_CFG    = Path("configs/steps/prediction.yaml")
SCORE_CFG   = Path("configs/steps/scoring.yaml")


def manual_walk_forward(
    df: pd.DataFrame,
    asof_closes_utc: pd.DatetimeIndex | list[pd.Timestamp],
    *,
    step_cfg: dict,
    close_et,
    run_id: str,
) -> pd.DataFrame:
    """
    Walk-forward 1-step-ahead forecasts across a set of as-of close timestamps (UTC).
    Returns a single DataFrame of forecasts for this run_id.
    """
    enabled_specs = get_enabled_model_specs(step_cfg)

    out = []
    asof_closes_utc = pd.DatetimeIndex(pd.to_datetime(asof_closes_utc, utc=True)).sort_values()

    for asof_close_utc in asof_closes_utc:
        # Use the as-of close as "now" for consistent walk-forward timing logic
        asof_date_et, asof_close_utc2, forecast_date_et, forecast_close_utc = resolve_prediction_times(
            now_utc=asof_close_utc.to_pydatetime(),
            close_et=close_et,
        )

        # Safety: keep a single source of truth
        asof_close_utc = pd.Timestamp(asof_close_utc2)

        # Train on data available up to this as-of timestamp
        df_asof = df.loc[df.index <= asof_close_utc]
        train_df = apply_train_window(df_asof, step_cfg)

        base_row = {
            "run_id": run_id,
            "asof_date_et": pd.Timestamp(asof_date_et).normalize(),
            "forecast_date_et": pd.Timestamp(forecast_date_et).normalize(),
            "asof_close_utc": pd.Timestamp(asof_close_utc),
            "forecast_close_utc": pd.Timestamp(forecast_close_utc),
        }

        preds = forecast_models(
            train_df=train_df,
            enabled_specs=enabled_specs,
            base_row=base_row,
        )

        if preds is None or len(preds) == 0:
            continue

        out.append(preds)

    if not out:
        # Return empty frame with no crash, caller can handle
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


def predict_backfill(
    storage,
    global_cfg: dict,
    step_cfg: dict,
    backfill_bdays: int = 252,
) -> pd.DataFrame:
    """
    Orchestrates 1-step-ahead variance forecasts for enabled models.
    Writes merged history to store_key, returns only this run's forecasts.
    """
    run_id = utc_run_id()

    in_key, store_key, close_et, returns_col, rv_col = parse_predict_cfg(step_cfg)

    df = load_and_standardize_modeling_table(
        storage=storage,
        in_key=in_key,
        returns_col=returns_col,
        rv_col=rv_col,
    )

    # Expectation: df.index is a UTC DatetimeIndex (your pipeline uses ET-close stamped in UTC)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected df.index to be a DatetimeIndex.")
    if df.index.tz is None:
        # If your table is UTC but tz-naive, force it (adjust if your actual semantics differ)
        df = df.copy()
        df.index = df.index.tz_localize("UTC")

    df = df.sort_index()

    # Resolve "current" as-of close so we don't backfill past what exists / what is considered available
    _, current_asof_close_utc, _, _ = resolve_prediction_times(
        now_utc=datetime.now(timezone.utc),
        close_et=close_et,
    )
    current_asof_close_utc = pd.Timestamp(current_asof_close_utc)

    # Choose walk-forward anchor timestamps from the data itself
    asof_closes_utc = df.index.unique().sort_values()
    asof_closes_utc = asof_closes_utc[asof_closes_utc <= current_asof_close_utc]

    if len(asof_closes_utc) == 0:
        return pd.DataFrame()

    asof_closes_utc = asof_closes_utc[-backfill_bdays:]

    results = manual_walk_forward(
        df,
        asof_closes_utc,
        step_cfg=step_cfg,
        close_et=close_et,
        run_id=run_id,
    )

    if results is None or len(results) == 0:
        return pd.DataFrame()

    merge_and_persist_predictions(
        storage=storage,
        store_key=store_key,
        results=results,
    )

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="How many days back to backfill")
    args = ap.parse_args()

    logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"), name="proj.backfill_pred_score")

    # Resolve paths relative to repo root / cwd
    run_all_cfg_path = RUN_ALL_CFG.resolve()
    pred_cfg_path = PRED_CFG.resolve()
    score_cfg_path = SCORE_CFG.resolve()

    logger.info("Using run_all cfg: %s", run_all_cfg_path)
    logger.info("Using prediction cfg: %s", pred_cfg_path)
    logger.info("Using scoring cfg: %s", score_cfg_path)

    cfg = load_config(run_all_cfg_path)
    storage = make_storage(cfg)

    pred_step_cfg = load_config(pred_cfg_path)
    score_step_cfg = load_config(score_cfg_path)

    predict_backfill(storage, cfg, pred_step_cfg, backfill_bdays=args.days)
    score_predictions(storage, cfg, score_step_cfg)
    


if __name__ == "__main__":
    main()
