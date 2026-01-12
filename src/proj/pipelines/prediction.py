# src/proj/pipelines/predict.py
from __future__ import annotations

import argparse
import os
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from proj.models.ewma import EWMAVariance
from proj.models.garch import GARCHModel
from proj.models.garchx import GARCHProxyX
from proj.models.harrv import HARRV
from proj.models.base import VolatilityModel
from proj.data.merge_helpers import merge_and_dedup_long
from proj.utils.dates import  *

import logging
logger = logging.getLogger("proj.prediction")  

ET = ZoneInfo("America/New_York")

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
        return EWMAVariance(lam=lam, input_type=input_type)

    if mtype == "garch":
        p = int(params.get("p", 1))
        q = int(params.get("q", 1))
        dist = str(params.get("dist", "t"))
        return GARCHModel(p=p, q=q, dist=dist)
    
    if mtype == 'garch-x':
        p = int(params.get("p", 1))
        q = int(params.get("q", 1))
        dist = str(params.get("dist", "t"))
        x_cols = params.get("x_cols", []) or []
        return GARCHProxyX(p=p, q=q, dist=dist, x_cols=x_cols)

    if mtype in {"har_rv", "harrv"}:
        x_cols = params.get("x_cols", []) or []
        return HARRV(x_cols=x_cols)

    raise ValueError(f"Unknown model type: {mtype}")

def compute_rolling_logrv_stats(
    gold: pd.DataFrame,
    realized_col: str,
    eps: float,
    window_bd: int = 60,
) -> pd.DataFrame:
    x = np.log(gold[realized_col].astype(float) + eps)

    mu = x.rolling(window_bd, min_periods=max(10, window_bd // 3)).mean()
    sd = x.rolling(window_bd, min_periods=max(10, window_bd // 3)).std(ddof=0)

    return pd.DataFrame(
        {
            "mu_logrv": mu,
            "sd_logrv": sd,
        },
        index=gold.index,
    )

def _require_k_consecutive(x: pd.Series, k: int = 3) -> pd.Series:
    out = x.copy()
    # whenever the value changes, start a new run
    run_id = (x != x.shift(1)).cumsum()
    run_len = x.groupby(run_id).cumcount() + 1
    # only accept the new regime once it has lasted k days; otherwise keep previous
    out[run_len < k] = np.nan
    out = out.ffill().fillna(x.iloc[0])
    return out



def add_forecasted_regime_from_gold(
    preds: pd.DataFrame,
    gold: pd.DataFrame,
    realized_col: str,
    eps: float = 1e-12,
    window_bd: int = 60,
    low_p: float = 0.30,
    high_p: float = 0.70,
) -> pd.DataFrame:
    if realized_col not in gold.columns:
        raise ValueError(f"gold missing '{realized_col}' for regime stats. Available: {list(gold.columns)}")
    if "predicted_value" not in preds.columns:
        raise ValueError("preds missing 'predicted_value'")

    stats = compute_rolling_logrv_stats(gold, realized_col=realized_col, eps=eps, window_bd=window_bd)

    p = preds.reset_index().rename(columns={preds.index.name or "index": "ts"})
    p["ts"] = pd.to_datetime(p["ts"], utc=True)
    p = p.sort_values("ts")

    s = stats.reset_index().rename(columns={stats.index.name or "index": "ts"})
    s["ts"] = pd.to_datetime(s["ts"], utc=True)
    s = s.sort_values("ts")

    p = pd.merge_asof(
        p, s,
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    )

    log_pred = np.log(p["predicted_value"].astype(float) + eps)
    z = (log_pred - p["mu_logrv"]) / p["sd_logrv"]

    p["regime_z_pred"] = z
    p["regime_pct_pred"] = norm.cdf(z).clip(0.0, 1.0)          # 0..1
    p["regime_pct_pred_100"] = 100.0 * p["regime_pct_pred"]    # 0..100
    
    p["regime_pred"] = np.select(
        [p["regime_pct_pred"] < low_p, p["regime_pct_pred"] > high_p],
        ["Low", "High"],
        default="Medium",
    )

    p["regime_pred_stable"] = _require_k_consecutive(p["regime_pred"], k=3)

    p = p.set_index("ts")
    p.index.name = preds.index.name or "forecast_close_utc"
    return p

# -----------------------
# Main prediction runner
# -----------------------
def predict_next(storage, global_cfg: dict, step_cfg: dict) -> pd.DataFrame:
    """
    Loads the modeling table (UTC), determines the as-of business day in ET,
    trains enabled models, and returns 1-step-ahead variance forecasts.

    Output includes both ET date labels and a UTC timestamp representation.
    """

    # =============================================================================
    # 1) Load config + set run identifiers
    # =============================================================================
    run_id = utc_run_id()

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

    # =============================================================================
    # 2) Load + validate data (keep timestamps in UTC)
    # =============================================================================
    df = storage.read_parquet(in_key)
    df = ensure_datetime_index_utc(df)

    missing = [c for c in (returns_col, rv_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Standardize names expected by models
    df = df.rename(columns={returns_col: "ret", rv_col: "rv"})

    # =============================================================================
    # 3) Resolve as-of + forecast targets (ET close -> UTC identity)
    # =============================================================================
    now_utc = datetime.now(timezone.utc)

    # As-of trading day label (ET) + as-of close timestamp (UTC)
    asof_date_et = resolve_asof_trade_date_et(now_utc, close_et=close_et)
    asof_close_utc = et_close_ts_utc(asof_date_et, close_et=close_et)

    # Forecast trading day label (ET) + forecast close timestamp (UTC) [JOIN KEY]
    forecast_date_et = next_business_day_et(asof_date_et)
    forecast_close_utc = et_close_ts_utc(forecast_date_et, close_et=close_et)

    # =============================================================================
    # 4) Build training frame (data available through as-of close)
    # =============================================================================
    df_asof = df[df.index <= asof_close_utc]
    train_df = apply_train_window(df_asof, step_cfg)

    # =============================================================================
    # 5) Fit models + collect forecasts (one row, model forecasts as columns)
    # =============================================================================
    model_specs = step_cfg.get("models", []) or []
    enabled_specs = [m for m in model_specs if m.get("enabled", True)]
    if not enabled_specs:
        raise ValueError("No enabled models found in step_cfg['models'].")

    factory_data_cfg = {"returns_col": "ret", "realized_var_col": "rv"}

    rows: list[dict[str, object]] = []

    base: dict[str, object] = {
        # identifiers
        "run_id": run_id,

        # labels (for humans / debugging)
        "asof_date_et": pd.Timestamp(asof_date_et).normalize(),
        "forecast_date_et": pd.Timestamp(forecast_date_et).normalize(),

        # identities (for joins)
        "asof_close_utc": pd.Timestamp(asof_close_utc),
        "forecast_close_utc": pd.Timestamp(forecast_close_utc),  # merge key vs gold.timestamp_utc
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

        model_id = spec.get("name") or model.name
        # one row per model per forecast date
        rows.append(
            {
                **base,
                "model": model_id,
                "predicted_value": var_hat,
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(by=["forecast_close_utc", 'model'])
        .set_index("forecast_close_utc")
        .sort_index()
        
    )



    # =============================================================================
    # 6) Merge + persist (dedupe on forecast_close_utc index)
    # =============================================================================
    if storage.exists(store_key):
        old_df = storage.read_parquet(store_key)
        merged = merge_and_dedup_long(old_df, results, "model")
    else:
        merged = results

    merged = add_forecasted_regime_from_gold(
        preds=merged,
        gold=df_asof,        # has column "rv" after renaming
        realized_col="rv",
    )

    storage.write_parquet(merged, store_key)

    # Return the single-run results (also with regime, if you want)
    results_with_regime = add_forecasted_regime_from_gold(
        preds=results,
        gold=df_asof,
        realized_col="rv",
    )

    return results_with_regime



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
