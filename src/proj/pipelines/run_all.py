import os 

from proj.utils.paths import *

from proj.pipelines.ingestion import ingest
from proj.pipelines.merge import merge_data
from proj.pipelines.build_features import construct_features
from proj.pipelines.training import train
from proj.pipelines.prediction import predict_next

def load_config(cfg_path):
    pass


def run_all(cfg_path: str):
    cfg = load_config(cfg_path)

    # ==== step 1 ====
    # ingest(cfg, cfg["step_configs"]["ingest"])
    # ==== step 2 ====
    # merge_data(cfg, cfg["step_configs"]["merge"])
    # ==== step 3 ====
    # construct_features(cfg, cfg["step_configs"]["features"])
    # ==== step 4 ====
    # train(cfg, cfg["step_configs"]["train"])
    # ==== step 5 ====
    # predict_next(cfg, cfg["step_configs"]["train"])



if __name__ == '__main__':

    ROOT = find_project_root(Path(__file__))
    PATHS = build_paths(ROOT)
    CONFIG = PATHS["CONFIG"]

    cfg_path = os.path.join(CONFIG, "run_all.yaml")

    run_all(cfg_path)
