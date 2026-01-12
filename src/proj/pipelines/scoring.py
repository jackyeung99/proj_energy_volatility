import numpy as np
import pandas as pd

from proj.evaluation.metrics import qlike_arr

import pandas as pd

import logging
logger = logging.getLogger("proj.scoring")  

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


    stats = compute_rolling_logrv_stats(
        gold=gold,
        realized_col=realized_col,
        eps=eps,
        window_bd=60,
    )

    stats_t = (
        stats.reset_index()
            .rename(columns={stats.index.name or "index": "ts"})
    )
    stats_t["ts"] = pd.to_datetime(stats_t["ts"], utc=True)
    stats_t = stats_t.sort_values("ts")

    # merge rolling stats into predictions using BACKWARD join
    preds_t = pd.merge_asof(
        preds_t,
        stats_t,
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(days=7),  # weekends / holidays safe
    )

    
    #forecasted regime (available even for last prediction)
    low_z, high_z = -0.75, 0.75

    log_pred = np.log(preds_t["predicted_value"].astype(float) + eps)
    z_pred = (log_pred - preds_t["mu_logrv"]) / preds_t["sd_logrv"]

    preds_t["regime_z_pred"] = z_pred
    preds_t["regime_pred"] = np.select(
        [z_pred < low_z, z_pred > high_z],
        ["Low", "High"],
        default="Medium",
    )

    #join realized volatility for scoring

    joined = pd.merge_asof(
        preds_t,
        gold_t[["ts", realized_col]],
        on="ts",
        direction="nearest",
        tolerance=tol,
    )

    # only rows with realized get QLIKE
    joined["is_scored"] = joined[realized_col].notna()

    scored = joined[joined["is_scored"]].copy()
    if scored.empty:
        logging.warning("Not enough data to score")
        return None

    scored["qlike"] = qlike_arr(
        scored[realized_col],
        scored["predicted_value"],
        eps=eps,
    )

    # ------------------------------------------------------------------
    # 4) rolling QLIKE (ignore unscored rows)
    # ------------------------------------------------------------------
    scored = scored.sort_values(["model", "ts"])

    scored["weekly_rolling_avg_qlike"] = (
        scored.groupby("model")["qlike"]
            .rolling(window=weekly_bd, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )

    scored["monthly_rolling_avg_qlike"] = (
        scored.groupby("model")["qlike"]
            .rolling(window=monthly_bd, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------------
    # 5) final output
    # ------------------------------------------------------------------
    out = scored.rename(columns={"ts": "forecasted_date"})[[
        "forecasted_date",
        "model",
        "predicted_value",
        realized_col,
        "qlike",
        "weekly_rolling_avg_qlike",
        "monthly_rolling_avg_qlike",
        "regime_pred",
        "regime_z_pred",
    ]].sort_values(["forecasted_date", "model"]).reset_index(drop=True)

    storage.write_parquet(out, out_path)
    return out
            
