# src/proj/pipelines/predict_utils.py
from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from proj.data.merge_helpers import merge_and_dedup_long  

from proj.utils.dates import (
    ensure_datetime_index_utc,
    resolve_asof_trade_date_et,
    et_close_ts_utc,
    next_trading_day_et,
)

from proj.models.factory import model_factory  

def apply_train_window(df: pd.DataFrame, step_cfg: dict) -> pd.DataFrame:
    tw = step_cfg.get("data", {}).get("train_window", {}) or {}
    mode = tw.get("mode", "rolling")
    min_obs = int(tw.get("min_obs", 400))

    if mode not in {"rolling", "expanding"}:
        raise ValueError("data.train_window.mode must be 'rolling' or 'expanding'.")

    if mode == "expanding":
        train = df
    else:
        rolling_days = int(tw.get("rolling_days", 2520))
        train = df.iloc[-rolling_days:]

    if len(train) < min_obs:
        raise ValueError(f"Not enough observations to fit models: have {len(train)}, need {min_obs}.")
    return train


def parse_predict_cfg(step_cfg: dict) -> tuple[str, str, time, str, str]:
    """
    Returns:
      in_key, store_key, close_et, returns_col, rv_col
    """
    data_cfg = step_cfg.get("data", {}) or {}
    run_cfg = step_cfg.get("run", {}) or {}

    in_key = data_cfg["data_dir"]
    store_key = data_cfg["store_path"]

    returns_col = data_cfg.get("returns_col", "ret")
    rv_col = data_cfg.get("realized_var_col", "rv")

    tz_name = run_cfg.get("timezone", "America/New_York")
    if tz_name != "America/New_York":
        raise ValueError(f"Unsupported timezone '{tz_name}'. Only America/New_York is supported.")

    close_str = run_cfg.get("market_close_et") or "16:00"
    hh, mm = (int(x) for x in close_str.split(":"))
    close_et = time(hh, mm)

    return in_key, store_key, close_et, returns_col, rv_col


def load_and_standardize_modeling_table(
    storage: Any,
    in_key: str,
    returns_col: str,
    rv_col: str,
) -> pd.DataFrame:
    """
    Loads modeling table and standardizes required columns to:
      - ret
      - rv
    Index is enforced UTC datetime.
    """
    df = storage.read_parquet(in_key)
    df = ensure_datetime_index_utc(df)

    missing = [c for c in (returns_col, rv_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    return df.rename(columns={returns_col: "ret", rv_col: "rv"})


def resolve_prediction_times(
    now_utc: datetime,
    close_et: time,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Returns:
      asof_date_et, asof_close_utc, forecast_date_et, forecast_close_utc
    """
    asof_date_et = pd.Timestamp(resolve_asof_trade_date_et(now_utc, close_et=close_et)).normalize()
    asof_close_utc = pd.Timestamp(et_close_ts_utc(asof_date_et, close_et=close_et))

    forecast_date_et = pd.Timestamp(next_trading_day_et(asof_date_et)).normalize()
    forecast_close_utc = pd.Timestamp(et_close_ts_utc(forecast_date_et, close_et=close_et))

    return asof_date_et, asof_close_utc, forecast_date_et, forecast_close_utc


def get_enabled_model_specs(step_cfg: dict) -> list[dict]:
    model_specs = step_cfg.get("models", []) or []
    enabled = [m for m in model_specs if m.get("enabled", True)]
    if not enabled:
        raise ValueError("No enabled models found in step_cfg['models'].")
    return enabled


def forecast_models(
    train_df: pd.DataFrame,
    enabled_specs: list[dict],
    base_row: dict,
) -> pd.DataFrame:
    """
    Fits each enabled model and returns a long table with one row per model.
    Expected output columns: base_row keys + model + predicted_value
    Index: forecast_close_utc
    """
    rows: list[dict] = []
    factory_data_cfg = {"returns_col": "ret", "realized_var_col": "rv"}

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
            raise ValueError(f"{model.name} produced invalid variance forecast: {var_hat}")

        model_id = spec.get("name") or model.name
        rows.append({**base_row, "model": model_id, "predicted_value": var_hat})

    out = (
        pd.DataFrame(rows)
        .sort_values(by=["forecast_close_utc", "model"])
        .set_index("forecast_close_utc")
        .sort_index()
    )
    return out


def merge_and_persist_predictions(storage: Any, store_key: str, results: pd.DataFrame) -> pd.DataFrame:
    """
    Merge current results into historical table (dedupe by forecast_close_utc + model),
    write back to store_key, return merged.
    """

    if storage.exists(store_key):
        old_df = storage.read_parquet(store_key)
        merged = merge_and_dedup_long(old_df, results, "model")
    else:
        merged = results

    storage.write_parquet(merged, store_key)
    return merged
