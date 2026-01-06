import os
from pathlib import Path

from proj.utils.paths import *
from proj.utils.config import load_config
from proj.data.storage import make_storage

from proj.pipelines.ingestion import ingest
from proj.pipelines.merge import merge_data
from proj.pipelines.build_features import construct_features
from proj.pipelines.prediction import predict_next


STEP_DISPATCH = {
    "ingestion": ingest,
    "features": construct_features,
    "merge": merge_data,
    "prediction": predict_next,
}


def run_all(cfg_path: str):
    cfg_path = Path(cfg_path).resolve()
    cfg = load_config(cfg_path)

    # Create storage once (local/cloud via config)
    storage = make_storage(cfg)

    steps = cfg.get("steps", {})

    for step_name, step_cfg_meta in steps.items():
        enabled = step_cfg_meta.get("enabled", True)
        step_cfg_path = step_cfg_meta.get("config")

        if not enabled:
            print(f"[SKIP] {step_name} (disabled)")
            continue

        if step_name not in STEP_DISPATCH:
            raise ValueError(f"Unknown pipeline step '{step_name}'")

        if not step_cfg_path:
            raise ValueError(f"Missing config path for step '{step_name}'")

        print(f"[RUN] {step_name}")

        step_cfg = load_config(step_cfg_path)
        step_fn = STEP_DISPATCH[step_name]

        step_fn(storage, cfg, step_cfg)

    print("[DONE] pipeline completed")


if __name__ == "__main__":
    ROOT = find_project_root(Path(__file__))
    PATHS = build_paths(ROOT)
    CONFIG = PATHS["CONFIG"]

    cfg_path = os.path.join(CONFIG, "run_all.yaml")
    run_all(cfg_path)
