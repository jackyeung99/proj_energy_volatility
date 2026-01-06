# src/proj/pipelines/predict.py
from __future__ import annotations

import argparse
import os
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from proj.models.ewma import EWMAVariance
from proj.models.garch import GARCHModel
from proj.models.harrv import HARRV
from proj.models.base import VolatilityModel
from proj.data.merge_helpers import merge_and_dedup

ET = ZoneInfo("America/New_York")


# -----------------------
# Helpers
# -----------------------
def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a sorted UTC DatetimeIndex. Do NOT convert storage to ET.
    """
    df = df.copy()
    df = df.sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("predict: dataframe index must be a DatetimeIndex")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


def resolve_asof_date_et(
    now_utc: datetime,
    market_close_et: time = time(16, 0),
) -> pd.Timestamp:
    """
    Decide the 'as-of' business date in Eastern time:
      - weekends -> previous business day
      - before market close -> previous business day
      - after close -> today (if weekday)
    Returns a normalized ET date timestamp (tz-naive pandas Timestamp representing date).
    """
    now_et = now_utc.astimezone(ET)
    today = pd.Timestamp(now_et.date())

    # weekend -> previous business day
    if today.dayofweek >= 5:
        return (today - pd.tseries.offsets.BDay(1)).normalize()

    # before close -> previous business day
    if now_et.time() < market_close_et:
        return (today - pd.tseries.offsets.BDay(1)).normalize()

    return today.normalize()


def et_date_to_utc_end(day_et: pd.Timestamp) -> pd.Timestamp:
    """
    End-of-day boundary in UTC for an ET calendar day: [start, end).
    We return 'end' so you can slice df.loc[:end_utc] safely using < end_utc.
    """
    start_et = datetime.combine(day_et.date(), time(0, 0), tzinfo=ET)
    end_et = start_et + pd.Timedelta(days=1)
    return pd.Timestamp(end_et.astimezone(timezone.utc))


def next_business_day_et(day_et: pd.Timestamp) -> pd.Timestamp:
    """
    Next business day label in ET (date).
    """
    return (day_et + pd.tseries.offsets.BDay(1)).normalize()


def forecast_label_to_utc_timestamp(day_et: pd.Timestamp) -> pd.Timestamp:
    """
    Represent a forecast *date label* (ET) as a UTC timestamp for output.
    Convention: UTC midnight of that date label.
    """
    return pd.Timestamp(day_et.date()).tz_localize("UTC")


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


def model_factory(model_spec: dict, data_cfg: dict) -> VolatilityModel:
    mtype = model_spec["type"].strip().lower()
    params = model_spec.get("params", {}) or {}

    returns_col = data_cfg.get("returns_col", "ret")

    if mtype == "ewma":
        lam = float(params.get("lam", 0.94))
        input_type = params.get("input_type", returns_col)
        return EWMAVariance(lam=.94, input_type=input_type)

    if mtype == "garch":
        p = int(params.get("p", 1))
        q = int(params.get("q", 1))
        dist = str(params.get("dist", "t"))
        return GARCHModel(p=p, q=q, dist=dist)

    if mtype in {"har_rv", "harrv"}:
        x_cols = params.get("x_cols", []) or []
        return HARRV(x_cols=x_cols)

    raise ValueError(f"Unknown model type: {mtype}")


# -----------------------
# Main prediction runner
# -----------------------
def predict_next(storage, global_cfg: dict, step_cfg: dict) -> pd.DataFrame:
    """
    Loads the modeling table (UTC), determines the as-of business day in ET,
    trains enabled models, and returns 1-step-ahead variance forecasts.

    Output includes both ET date labels and a UTC timestamp representation.
    """
    run_id = utc_run_id()

    data_cfg = step_cfg.get("data", {}) or {}
    run_cfg = step_cfg.get("run", {}) or {}

    in_key = data_cfg["data_dir"]
    store_key = data_cfg["store_path"]
    returns_col = data_cfg.get("returns_col", "ret")
    rv_col = data_cfg.get("realized_var_col", "rv")

    # ---- Load data (keep UTC)
    df = storage.read_parquet(in_key)
    df = ensure_datetime_index(df)

    # ---- Validate columns
    missing = [c for c in [returns_col, rv_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}. Available: {list(df.columns)}")

    # standardize
    df = df.rename(columns={rv_col: "rv", returns_col: 'ret'})

    # ---- Decide as-of date in ET based on current time
    tz_name = run_cfg.get("timezone", "America/New_York")
    if tz_name != "America/New_York":
        # If you ever extend, support other tz via ZoneInfo
        pass

    now_utc = datetime.now(timezone.utc)
    close_str = (run_cfg.get("market_close_et") or "16:00")
    hh, mm = [int(x) for x in close_str.split(":")]
    asof_date_et = resolve_asof_date_et(now_utc, market_close_et=time(hh, mm))

    # Slice data up through end of as-of ET day (converted to UTC boundary)
    end_utc = et_date_to_utc_end(asof_date_et)
    df_asof = df[df.index < end_utc]

    # ---- Training window
    train_df = apply_train_window(df_asof, step_cfg)

    # ---- Forecast labels
    forecast_date_et = next_business_day_et(asof_date_et)
    forecast_ts_utc = forecast_label_to_utc_timestamp(forecast_date_et)

    # ---- Results accumulator
    results = pd.DataFrame(
        columns=[
            "run_id",
            "asof_date_et",
            "forecast_date_et",
            "asof_end_utc",
            "timestamp",  # UTC representation of forecast day
            "model",
            "variance_forecast",
        ]
    )

    # ---- Fit + forecast enabled models
    model_specs = step_cfg.get("models", []) or []
    enabled_specs = [m for m in model_specs if m.get("enabled", True)]
    if not enabled_specs:
        raise ValueError("No enabled models found in step_cfg['models'].")

    factory_data_cfg = {
        "returns_col": returns_col,
        "realized_var_col": "rv",
    }

    # ---- Global results: 1 row per run, model forecasts as columns
    row = {
        "run_id": run_id,

        # metadata (date labels, not identity)
        "asof_date_et": pd.Timestamp(asof_date_et).normalize(),

        # identity (will become the index later)
        "forecast_date_et": pd.Timestamp(forecast_date_et).normalize(),

        # optional audit/debug (keep, but not used for joins)
        "asof_end_utc": pd.Timestamp(end_utc),
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
            raise ValueError(f"{model.name} produced invalid variance forecast: {var_hat}")

        col = spec.get("name") or model.name

        # prevent accidental overwrite if two models share same name
        if col in row:
            raise ValueError(f"Duplicate model column name '{col}'. Give models unique 'name' in config.")

        row[col] = var_hat

    # one-row dataframe
    results = pd.DataFrame([row])
    results = (
        results
        .set_index("forecast_date_et")
        .sort_index()
    )

    if storage.exists(store_key):
        old_df = storage.read_parquet(store_key)
        merged = merge_and_dedup(old_df, results)
    else:
        merged = results


    storage.write_parquet(merged, store_key)


    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=False, default=os.environ.get("CFG_PATH"))
    args = parser.parse_args()

    if not args.cfg:
        raise ValueError("Provide --cfg <path> or set CFG_PATH env var.")

    with open(args.cfg, "r") as f:
        step_cfg = yaml.safe_load(f)

    # This CLI is mainly for local testing; in the pipeline, run_all passes storage + cfg.
    # If you want CLI execution, you can wire storage creation here.
    raise SystemExit(
        "predict.py is intended to be called from run_all with (storage, global_cfg, step_cfg). "
        "Run via your run_all pipeline, or add CLI storage wiring if desired."
    )


if __name__ == "__main__":
    main()
