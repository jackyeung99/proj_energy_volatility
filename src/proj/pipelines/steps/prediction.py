# src/proj/pipelines/predict.py
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from proj.utils.dates import utc_run_id

from proj.pipelines.steps._predict_utils import (
    parse_predict_cfg,
    apply_train_window,
    load_and_standardize_modeling_table,
    resolve_prediction_times,
    get_enabled_model_specs,
    forecast_models,
    merge_and_persist_predictions,
)

def predict_next(storage, global_cfg: dict, step_cfg: dict) -> pd.DataFrame:
    """
    Orchestrates 1-step-ahead variance forecasts for enabled models.
    Writes merged history to store_path, returns only this run's forecasts.
    """
    run_id = utc_run_id()

    in_key, store_key, close_et, returns_col, rv_col = parse_predict_cfg(step_cfg)

    df = load_and_standardize_modeling_table(
        storage=storage,
        in_key=in_key,
        returns_col=returns_col,
        rv_col=rv_col,
    )

    asof_date_et, asof_close_utc, forecast_date_et, forecast_close_utc = resolve_prediction_times(
        now_utc=datetime.now(timezone.utc),
        close_et=close_et,
    )

    df_asof = df[df.index <= asof_close_utc]
    train_df = apply_train_window(df_asof, step_cfg)

    enabled_specs = get_enabled_model_specs(step_cfg)

    base_row = {
        "run_id": run_id,
        "asof_date_et": pd.Timestamp(asof_date_et).normalize(),
        "forecast_date_et": pd.Timestamp(forecast_date_et).normalize(),
        "asof_close_utc": pd.Timestamp(asof_close_utc),
        "forecast_close_utc": pd.Timestamp(forecast_close_utc),
    }

    results = forecast_models(
        train_df=train_df,
        enabled_specs=enabled_specs,
        base_row=base_row,
    )

    merge_and_persist_predictions(
        storage=storage,
        store_key=store_key,
        results=results,
    )

    return results
