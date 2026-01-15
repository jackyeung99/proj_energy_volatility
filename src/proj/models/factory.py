# src/proj/models/factory.py
from __future__ import annotations

from proj.models.volatility_models.base import VolatilityModel
from proj.models.volatility_models.ewma import EWMAVariance
from proj.models.volatility_models.garch import GARCHModel
from proj.models.volatility_models.garchx import GARCHProxyX
from proj.models.volatility_models.harrv import HARRV


def model_factory(model_spec: dict, data_cfg: dict) -> VolatilityModel:
    """
    Construct a volatility model from config.
    This is the ONLY place that knows how model types map to classes.
    """
    mtype = model_spec["type"].strip().lower()
    params = model_spec.get("params", {}) or {}

    returns_col = data_cfg.get("returns_col", "ret")

    if mtype == "ewma":
        return EWMAVariance(
            lam=float(params.get("lam", 0.94)),
            input_type=params.get("input_type", returns_col),
        )

    if mtype == "garch":
        return GARCHModel(
            p=int(params.get("p", 1)),
            q=int(params.get("q", 1)),
            dist=str(params.get("dist", "t")),
        )

    if mtype in {"garch-x", "garchx"}:
        return GARCHProxyX(
            p=int(params.get("p", 1)),
            q=int(params.get("q", 1)),
            dist=str(params.get("dist", "t")),
            x_cols=params.get("x_cols", []) or [],
        )

    if mtype in {"har_rv", "harrv"}:
        return HARRV(
            x_cols=params.get("x_cols", []) or [],
        )

    raise ValueError(f"Unknown model type: {mtype}")
