import numpy as np
import pandas as pd

from proj.evaluation.metrics import qlike_arr

import pandas as pd

import logging
logger = logging.getLogger("proj.scoring")  


def score_predictions(storage, global_cfg: dict, step_cfg: dict) -> pd.DataFrame:
    files_cfg = step_cfg["files"]
    gold_path = files_cfg["rv_data_dir"]
    pred_path = files_cfg["predictions_dir"]
    out_path  = files_cfg["output_dir"]

    realized_col = step_cfg.get("realized_col", "rv_xle")
    eps          = float(step_cfg.get("eps", 1e-12))

    aggregation_cfg = step_cfg.get("aggregation", {}) or {}
    metrics = aggregation_cfg.get("metrics", ["qlike"])

    weekly_bd  = int(step_cfg.get("weekly_window_bd", 5))
    monthly_bd = int(step_cfg.get("monthly_window_bd", 21))

    # --- load ---
    gold  = storage.read_parquet(gold_path)
    preds = storage.read_parquet(pred_path)


    if gold is None or len(gold) == 0:
        raise ValueError(f"gold is empty at {gold_path}")

    if preds is None or len(preds) == 0:
        logging.warning("Not enough Predictions to score")
        return 

    # --- index hygiene ---
    gold = gold.copy()
    gold.index = pd.to_datetime(gold.index, utc=True)

    preds = preds.copy()
    preds.index = pd.to_datetime(preds.index, utc=True)

    if realized_col not in gold.columns:
        raise ValueError(f"gold missing '{realized_col}'. Available: {list(gold.columns)}")
    if "predicted_value" not in preds.columns:
        raise ValueError(f"preds missing 'predicted_value'. Available: {list(preds.columns)}")
    if "model" not in preds.columns:
        raise ValueError("preds missing required column 'model'")


    # --- join: robust time join (index -> column + merge_asof tolerance) ---
    tol = pd.Timedelta(hours=2)

    preds_t = preds.reset_index().rename(columns={preds.index.name or "index": "ts"})
    gold_t  = gold.reset_index().rename(columns={gold.index.name  or "index": "ts"})

    preds_t["ts"] = pd.to_datetime(preds_t["ts"], utc=True)
    gold_t["ts"]  = pd.to_datetime(gold_t["ts"],  utc=True)

    preds_t = preds_t.sort_values("ts")
    gold_t  = gold_t.sort_values("ts")

    joined = pd.merge_asof(
        preds_t,
        gold_t[["ts", realized_col]],
        on="ts",
        direction="nearest",   # change to "backward" if you only want earlier realized values
        tolerance=tol,
    )

    joined = joined.dropna(subset=[realized_col])
    if joined.empty:
        logging.warning("Not enough data to score")
        return None

    joined["qlike"] = qlike_arr(joined[realized_col], joined["predicted_value"], eps=eps)

    joined = joined.sort_values(["model", "ts"])

    joined["weekly_rolling_avg_qlike"] = (
        joined.groupby("model")["qlike"]
            .rolling(window=weekly_bd, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )

    joined["monthly_rolling_avg_qlike"] = (
        joined.groupby("model")["qlike"]
            .rolling(window=monthly_bd, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )   


    out = joined.rename(columns={"ts": "forecasted_date"})[[
        "forecasted_date",
        "model",
        "predicted_value",
        "rv_xle",
        "qlike",
        "weekly_rolling_avg_qlike",
        "monthly_rolling_avg_qlike",
    ]].sort_values(["forecasted_date", "model"]).reset_index(drop=True)

    storage.write_parquet(out, out_path)
    return out
            
