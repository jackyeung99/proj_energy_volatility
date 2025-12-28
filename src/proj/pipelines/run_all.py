import os 

from proj.utils.paths import *
from proj.utils.config import load_config
from proj.data.storage import LocalStorage, URIStorage

from proj.pipelines.ingestion import ingest
from proj.pipelines.merge import merge_data
from proj.pipelines.build_features import construct_features
from proj.pipelines.training import train
from proj.pipelines.prediction import predict_next

def make_storage(global_cfg: dict):
    s = global_cfg["storage"]
    if s["backend"] == "local":
        return LocalStorage(base_dir=Path(s["base_dir"]))
    if s["backend"] == "uri":
        return URIStorage(base_uri=s["base_uri"])
    raise ValueError(f"Unknown storage backend: {s['backend']}")


def run_all(cfg_path: str):
    run_cfg_path = Path(cfg_path).resolve()
    cfg = load_config(cfg_path)
    print(cfg)

     # Create storage once (toggle local/cloud via config)
    storage = make_storage(cfg)

    # Step config paths are defined in cfg["steps"]
    steps = cfg.get("steps", {})

   # ==== step 1 ====
    ingest_cfg_path = steps["ingestion"]
    ingest_cfg = load_config(ingest_cfg_path)
    # ingest(storage, cfg, ingest_cfg)

    # ==== step 2 ====
    # merge_cfg_path = resolve_step_path(run_cfg_path, steps["merge"])
    # merge_cfg = load_config(merge_cfg_path)
    # merge_data(storage, cfg, merge_cfg)

    # ==== step 3 ====
    # features_cfg_path = resolve_step_path(run_cfg_path, steps["features"])
    # features_cfg = load_config(features_cfg_path)
    # construct_features(storage, cfg, features_cfg)

    # ==== step 4 ====
    # train_cfg_path = resolve_step_path(run_cfg_path, steps["train"])
    # train_cfg = load_config(train_cfg_path)
    # train(storage, cfg, train_cfg)

    # ==== step 5 ====
    # pred_cfg_path = resolve_step_path(run_cfg_path, steps["prediction"])
    # pred_cfg = load_config(pred_cfg_path)
    # predict_next(storage, cfg, pred_cfg)



if __name__ == '__main__':

    ROOT = find_project_root(Path(__file__))
    PATHS = build_paths(ROOT)
    CONFIG = PATHS["CONFIG"]

    cfg_path = os.path.join(CONFIG, "run_all.yaml")
    run_all(cfg_path)
