
from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(cfg_path: str | Path) -> Dict[str, Any]:
    """
    Load a configuration file (YAML or TOML).

    Parameters
    ----------
    cfg_path : str or Path
        Path to config file (.yaml, .yml, or .toml)

    Returns
    -------
    dict
        Parsed configuration dictionary
    """
    cfg_path = Path(cfg_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config format '{suffix}'. "
            "Use .yaml, .yml, or .toml."
        )

    if cfg is None:
        raise ValueError(f"Config file {cfg_path} is empty.")

    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a dictionary.")

    return cfg