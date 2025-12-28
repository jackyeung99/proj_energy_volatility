import os 

from proj.utils.paths import *
from proj.utils.config import load_config

from proj.pipelines.ingestion import ingest
from proj.pipelines.merge import merge_data
from proj.pipelines.build_features import construct_features
from proj.pipelines.training import train
from proj.pipelines.prediction import predict_next




def run_all(cfg_path: str):
    cfg = load_config(cfg_path)
    print(cfg)

    # ==== step 1 ====
    ingest_cfg = load_config(cfg['steps']['ingestion'])
    # ingest(cfg, cfg["step_configs"]["ingest"])

    # ==== step 2 ====
    # merge_cfg = load_config(cfg['steps']['merge'])
    # merge_data(cfg, cfg["step_configs"]["merge"])

    # ==== step 3 ====
    # features_cfg = load_config(cfg['steps']['features'])
    # construct_features(cfg, cfg["step_configs"]["features"])

    # ==== step 4 ====
    # train_cfg = load_config(cfg['steps']['train'])
    # train(cfg, cfg["step_configs"]["train"])

    # ==== step 5 ====
    # predict_cfg = load_config(cfg['steps']['prediction'])
    # predict_next(cfg, cfg["step_configs"]["train"])



if __name__ == '__main__':

    ROOT = find_project_root(Path(__file__))
    PATHS = build_paths(ROOT)
    CONFIG = PATHS["CONFIG"]

    cfg_path = os.path.join(CONFIG, "run_all.yaml")
    run_all(cfg_path)
