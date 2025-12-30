import os 

from proj.utils.paths import *
from proj.utils.config import load_config
from proj.data.storage import make_storage

from proj.pipelines.ingestion import ingest
from proj.pipelines.merge import merge_data
from proj.pipelines.build_features import construct_features
from proj.pipelines.training import train
from proj.pipelines.prediction import predict_next



def run_all(cfg_path: str):
    run_cfg_path = Path(cfg_path).resolve()
    cfg = load_config(cfg_path)

    # Create storage once (toggle local/cloud via config)
    storage = make_storage(cfg)

    steps = cfg.get("steps", {})

    # ==== Step 1: Ingestion ====
    if "ingestion" in steps:
        ingest_cfg = load_config(steps["ingestion"])
        if ingest_cfg.get("enabled", True):
            ingest(storage, cfg, ingest_cfg)
        else:
            print("Skipping ingestion (disabled)")

    # ==== Step 2: Feature construction ====
    if "features" in steps:
        features_cfg = load_config(steps["features"])
        if features_cfg.get("enabled", True):
            construct_features(storage, cfg, features_cfg)
        else:
            print("Skipping features (disabled)")

    # ==== Step 3: Merge ====
    if "merge" in steps:
        merge_cfg = load_config(steps["merge"])
        if merge_cfg.get("enabled", True):
            merge_data(storage, cfg, merge_cfg)
        else:
            print("Skipping merge (disabled)")

    # ==== Step 4: Prediction (optional) ====
    if "prediction" in steps:
        pred_cfg = load_config(steps["prediction"])
        if pred_cfg.get("enabled", False):
            predict_next(storage, cfg, pred_cfg)
        else:
            print("Skipping prediction (disabled)")



if __name__ == '__main__':

    ROOT = find_project_root(Path(__file__))
    PATHS = build_paths(ROOT)
    CONFIG = PATHS["CONFIG"]

    cfg_path = os.path.join(CONFIG, "run_all.yaml")
    run_all(cfg_path)
