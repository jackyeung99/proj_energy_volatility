# src/proj/pipelines/merge_gold.py
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from proj.data.merge_helpers import MergeSpec, merge_to_gold


def merge_data(storage, global_cfg: dict, step_cfg: dict) -> dict:
    """
    Merge multiple SILVER datasets into one GOLD modeling table.

    Assumptions:
      - storage.read(path) -> pd.DataFrame
      - storage.write(path, df) -> None
      - step_cfg["merge_gold"] contains inputs/output/merge settings

    Returns:
      dict with metadata (rows/cols/output_path)
    """
    cfg = step_cfg["merge_gold"]

    # -------------------------
    # Load inputs
    # -------------------------
    inputs: Dict[str, str] = cfg["inputs"]
    datasets: Dict[str, pd.DataFrame] = {}

    for name, spec in inputs.items():
        enabled = spec.get("enabled", True)
        path = spec.get("path")

        if not enabled:
            print(f"    [SKIP INPUT] {name}")
            continue


        df = storage.read_parquet(path)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"merge_gold: input '{name}' at '{path}' is empty or not a DataFrame")
        datasets[name] = df

    # -------------------------
    # Build merge spec
    # -------------------------
    merge_cfg = cfg.get("merge", {})
    spec = MergeSpec(
        anchor_name=merge_cfg.get("anchor", "equities"),
        join_how=merge_cfg.get("how", "left"),
        dropna_target=bool(merge_cfg.get("dropna_target", True)),
        target_cols=tuple(merge_cfg.get("target_cols", ["log_rv_idio"])),
        start_date=merge_cfg.get("start_date"),
        end_date=merge_cfg.get("end_date"),
        enforce_lags_only=bool(merge_cfg.get("enforce_lags_only", True)),
        prefixes=merge_cfg.get("prefixes"),
    )

    # -------------------------
    # Merge
    # -------------------------
    gold = merge_to_gold(datasets, spec)

    # -------------------------
    # Optional postprocess
    # -------------------------
    post = cfg.get("postprocess", {})
    drop_cols = post.get("drop_cols", []) or []
    if drop_cols:
        gold = gold.drop(columns=[c for c in drop_cols if c in gold.columns])

    # -------------------------
    # Write output
    # -------------------------
    output_path = cfg["output"]["store_path"]
    storage.write_parquet(gold, output_path)

    return {
        "enabled": True,
        "output_path": output_path,
        "rows": int(gold.shape[0]),
        "cols": int(gold.shape[1]),
        "columns": list(gold.columns),
    }
