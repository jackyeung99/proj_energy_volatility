from __future__ import annotations

import os
import argparse
from pathlib import Path

import pandas as pd
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from proj.models.ewma import EWMAVariance
from proj.models.garch import GARCHModel
from proj.models.garchx import GARCHProxyX
from proj.models.harrv import HARRV
from proj.models.base import VolatilityModel
from proj.data.merge_helpers import merge_and_dedup_long
from proj.utils.dates import  *

from proj.utils.config import load_config
from proj.utils.logging import setup_logging, log_step
from proj.data.storage import make_storage

from proj.pipelines.prediction import apply_train_window, model_factory
from proj.pipelines.scoring import score_predictions


from typing import Optional

# ----------------------------
# HARDCODED CONFIG PATHS
# ----------------------------
RUN_ALL_CFG = Path("configs/run_all.yaml")
PRED_CFG    = Path("configs/steps/prediction.yaml")
SCORE_CFG   = Path("configs/steps/scoring.yaml")


def next_trading_day(index: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Return the next timestamp in `index` after `ts`.
    Works even if `ts` is not exactly contained in the index (uses searchsorted).
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    index = index.sort_values()

    ts = pd.Timestamp(ts)
    # Ensure comparable tz
    if index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(index.tz)
    elif index.tz is None and ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    elif index.tz is not None and ts.tzinfo is not None and index.tz != ts.tzinfo:
        ts = ts.tz_convert(index.tz)

    # find insertion position strictly after ts
    pos = index.searchsorted(ts, side="right")
    if pos >= len(index):
        return None
    return index[pos]


def predict_backfill(
    storage,
    global_cfg: dict,
    step_cfg: dict,
    backfill_bdays: int = 252,
    end_date_et: Optional[str] = None,  # "YYYY-MM-DD" in ET
) -> pd.DataFrame:
    """
    Backfill 1-step-ahead variance forecasts for last N *available trading days*.

    For each as-of ET close timestamp t_et in the dataset:
      - train on data through t_close_utc
      - forecast for the next available trading close timestamp in the dataset
    """

    # =============================================================================
    # 1) Load config
    # =============================================================================
    run_id = utc_run_id()

    data_cfg = step_cfg.get("data", {}) or {}
    run_cfg = step_cfg.get("run", {}) or {}

    in_key = data_cfg["data_dir"]
    store_key = data_cfg["store_path"]

    returns_col = data_cfg.get("returns_col", "ret")
    rv_col = data_cfg.get("realized_var_col", "rv")

    close_str = run_cfg.get("market_close_et") or "16:00"
    hh, mm = (int(x) for x in close_str.split(":"))
    close_et = time(hh, mm)

    # =============================================================================
    # 2) Load + validate data (UTC index)
    # =============================================================================
    df = storage.read_parquet(in_key)
    df = ensure_datetime_index_utc(df).sort_index()

    missing = [c for c in (returns_col, rv_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    df = df.rename(columns={returns_col: "ret", rv_col: "rv"})

    bad = df[["ret", "rv"]].isna().any()
    print("ret/rv NaNs?", bad)

    X_cols = ["etf_USO_absret", "log_rv_spy", "wx_wind_speed_10m_mean_anom_z"] # whatever GARCHProxyX uses
    print("X NaNs?", df[X_cols].isna().sum().sort_values(ascending=False).head(10))
    print("Any inf?", np.isinf(df[X_cols]).any().any())

    # =============================================================================
    # 3) Build the ET-close trading index from the data itself
    # =============================================================================
    df_index_et = df.index.tz_convert("America/New_York")

    # This is your "trading calendar": ET-close timestamps actually present in the data.
    asof_index_et = pd.DatetimeIndex(df_index_et).sort_values()

    if asof_index_et.empty:
        raise ValueError("No rows in modeling table.")

    # Handle end_date_et: interpret as "that day's market close"
    if end_date_et is not None:
        end_close_et = (
            pd.Timestamp(end_date_et)
            .tz_localize("America/New_York")
            .normalize()
            .replace(hour=close_et.hour, minute=close_et.minute)
        )
    else:
        end_close_et = asof_index_et.max()

    # Keep only trading closes up to end_close_et
    asof_index_et = asof_index_et[asof_index_et <= end_close_et]
    if asof_index_et.empty:
        raise ValueError("No available trading closes <= end_date_et.")

    # Take last N available trading days
    asof_dates_et = asof_index_et[-backfill_bdays:]

    # =============================================================================
    # 4) Prepare models
    # =============================================================================
    model_specs = step_cfg.get("models", []) or []
    enabled_specs = [m for m in model_specs if m.get("enabled", True)]
    if not enabled_specs:
        raise ValueError("No enabled models found in step_cfg['models'].")

    factory_data_cfg = {"returns_col": "ret", "realized_var_col": "rv"}

    rows: list[dict[str, object]] = []

    # =============================================================================
    # 5) Walk-forward loop
    # =============================================================================
    for asof_close_et in asof_dates_et:
        forecast_close_et = next_trading_day(asof_index_et, asof_close_et)
        if forecast_close_et is None:
            continue

        asof_close_utc = asof_close_et.tz_convert("UTC")
        forecast_close_utc = forecast_close_et.tz_convert("UTC")

        # train data through as-of close (compare in UTC)
        df_asof = df[df.index <= asof_close_utc]
        train_df = apply_train_window(df_asof, step_cfg)
        print(train_df.iloc[-1]) 

        base = {
            "run_id": run_id,
            "asof_date_et": asof_close_et.normalize(),         # ET date label
            "forecast_date_et": forecast_close_et.normalize(), # ET date label
            "asof_close_utc": pd.Timestamp(asof_close_utc),
            "forecast_close_utc": pd.Timestamp(forecast_close_utc),
        }

    
        for spec in enabled_specs:
            model = model_factory(spec, factory_data_cfg)
            model.fit(train_df)

            fc = model.forecast(train_df)
            if not isinstance(fc, pd.Series):
                raise TypeError(f"{model.name}.forecast must return pd.Series, got {type(fc)}.")
            if len(fc) != 1:
                raise ValueError(f"{model.name}.forecast must return exactly 1 value; got {len(fc)}.")

            var_hat = float(fc.iloc[0])
            if not np.isfinite(var_hat) or var_hat <= 0:
                print(train_df)
                raise ValueError(f"{model.name} produced invalid variance forecast: {var_hat}")

            model_id = spec.get("name") or model.name
            rows.append({**base, "model": model_id, "predicted_value": var_hat})


    results = (
        pd.DataFrame(rows)
        .sort_values(by=["forecast_close_utc", "model"])
        .set_index("forecast_close_utc")
        .sort_index()
    )

    print(results)
    # =============================================================================
    # 6) Merge + persist (idempotent)
    # =============================================================================
    # if storage.exists(store_key):
    #     old_df = storage.read_parquet(store_key)
    #     merged = merge_and_dedup_long(old_df, results, "model")
    # else:
    #     merged = results

    # storage.write_parquet(merged, store_key)
    # return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180, help="How many days back to backfill")
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
    


if __name__ == "__main__":
    main()
