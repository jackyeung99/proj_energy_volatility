from __future__ import annotations

from typing import Any, Dict
import pandas as pd 
import numpy as np

from proj.data.ingestion_state import get_last_available_date, update_state
from proj.features import transforms
from proj.features.preprocessing import preprocess_equities, preprocess_equities_daily, preprocess_weather
from proj.data.storage import Storage


import logging
logger = logging.getLogger("proj.build_features")  



def preprocess_by_source(source_name: str, df: pd.DataFrame, base_features_cfg: dict,  source_cfg: dict):
    if source_name == "equities_intra":
        return preprocess_equities(df, base_features_cfg,  source_cfg)
    if source_name == "equities_daily" or source_name == "macro":
        return preprocess_equities_daily(df, base_features_cfg,  source_cfg)
    if source_name == "weather":
        return preprocess_weather(df, base_features_cfg,  source_cfg)
    raise ValueError(f"Unknown source '{source_name}'")



def preprocess_one_source(
        storage: Storage,
        base_features_cfg: dict,
        source_name: str,
        source_cfg: dict,
) -> dict:

    # retrieve raw data 
    data_key = source_cfg["data_path"]
    raw_df = storage.read_parquet(data_key)

    #perform transformations and pre processing
    new_df = preprocess_by_source(source_name, raw_df, base_features_cfg, source_cfg)

    logging.info(
        "SOURCE %s | length of preprocessed data %s",
        source_name,
        len(new_df)
    )
    


    store_key = source_cfg["store_path"]
    storage.write_parquet(new_df, store_key)


    return {
        "source": source_name,
        "store_key": store_key,
        "rows_written": int(len(raw_df)),
    }
    


def construct_features(storage: Storage, global_cfg: dict, step_cfg: dict) -> dict:

    sources_cfg = step_cfg["sources"]

    results: Dict[str, Any] = {}


    # Explicit per-source calls
    eq_cfg = sources_cfg.get("equities_intra", {})
    if eq_cfg.get("enabled", True):
        results["equities_intra"] = preprocess_one_source(storage, step_cfg, "equities_intra", eq_cfg)

    eq_daily_cfg = sources_cfg.get("equities_daily", {})
    if eq_daily_cfg.get("enabled", True):
        results["equities_daily"] = preprocess_one_source(storage, step_cfg, "equities_daily", eq_daily_cfg)

    macro_cfg = sources_cfg.get("macro", {})
    if macro_cfg.get("enabled", True):
        results["macro"] = preprocess_one_source(storage, step_cfg, "macro", macro_cfg)

    weather_cfg = sources_cfg.get("weather", {})
    if weather_cfg.get("enabled", True):
        results["weather"] = preprocess_one_source(storage, step_cfg, "weather", weather_cfg)

    return {"feature_results": results}

