from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


from proj.utils.config import load_config
from proj.utils.logging import setup_logging, log_step
from proj.data.storage import make_storage

from proj.pipelines.steps.ingestion import ingest
from proj.pipelines.steps.merge import merge_data
from proj.pipelines.steps.build_features import construct_features
from proj.pipelines.steps.prediction import predict_next
from proj.pipelines.steps.scoring import score_predictions


STEP_DISPATCH = {
    "ingestion": ingest,
    "features": construct_features,
    "merge": merge_data,
    "prediction": predict_next,
    "scoring": score_predictions, 
}


def run_all(cfg_path: str) -> None:
    logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"), name="proj.run_all")

    cfg_path = Path(cfg_path).resolve()
    logger.info("Loading config: %s", cfg_path)

    cfg = load_config(cfg_path)

    # Create storage once (local/cloud via config)
    storage = make_storage(cfg)

    # Helpful startup context (shows up in CloudWatch)
    s = cfg.get("storage", {})
    logger.info(
        "Storage backend=%s bucket=%s prefix=%s base_dir=%s",
        s.get("backend"),
        s.get("bucket"),
        s.get("prefix"),
        s.get("base_dir"),
    )

    steps = cfg.get("steps", {})
    if not steps:
        raise ValueError("No steps found in cfg['steps']. Check your run_all config.")

    for step_name, step_cfg_meta in steps.items():
        enabled = step_cfg_meta.get("enabled", True)
        step_cfg_path = step_cfg_meta.get("config")

        if not enabled:
            logger.info("STEP_SKIP %s (disabled)", step_name)
            continue

        if step_name not in STEP_DISPATCH:
            raise ValueError(f"Unknown pipeline step '{step_name}'. Known: {list(STEP_DISPATCH)}")

        if not step_cfg_path:
            raise ValueError(f"Missing config path for step '{step_name}'")

        step_fn = STEP_DISPATCH[step_name]

        # Make step config path relative to the run_all config directory if needed
        step_cfg_path = Path(step_cfg_path)
        if not step_cfg_path.is_absolute():
            step_cfg_path = (cfg_path.parent / step_cfg_path).resolve()

        with log_step(logger, step_name):
            logger.info("Loading step config: %s", step_cfg_path)
            step_cfg = load_config(step_cfg_path)

            logger.debug("Step '%s' config keys: %s", step_name, sorted(step_cfg.keys()))
            step_fn(storage, cfg, step_cfg)

    logger.info("DONE pipeline completed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        required=False,
        default=os.environ.get("CFG_PATH"),
        help="Path to run_all YAML config. If omitted, uses CFG_PATH env var.",
    )
    args = parser.parse_args()

    if not args.cfg:
        raise ValueError("Provide --cfg <path> or set CFG_PATH env var.")

    # Optional: “prove entrypoint” logs (enable with DEBUG_ENTRYPOINT=1)
    if os.getenv("DEBUG_ENTRYPOINT") == "1":
        print(">>> RUN_ALL MAIN STARTED <<<")
        print("ARGV:", sys.argv)
        print("CFG_PATH env:", os.getenv("CFG_PATH"))
        print("--cfg arg:", args.cfg)

    run_all(args.cfg)


if __name__ == "__main__":
    main()
