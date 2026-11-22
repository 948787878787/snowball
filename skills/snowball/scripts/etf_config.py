"""Config loading and validation for the core ETF tracker (`configs/core_etf.json`)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

CASH = "CASH"
_TW_SUFFIX = re.compile(r"\.(TW|TWO)$")


def is_usd(symbol: str) -> bool:
    """Symbols without a Taiwan exchange suffix are treated as USD-priced (converted with TWD=X)."""
    return symbol != CASH and not _TW_SUFFIX.search(symbol)


def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    targets = cfg.get("targets", {})
    if CASH not in targets:
        raise ValueError("targets must include a CASH line (cash buffer)")
    if any(w < 0 for w in targets.values()):
        raise ValueError("target weights must be >= 0")
    if abs(sum(targets.values()) - 1.0) > 1e-9:
        raise ValueError(f"target weights must sum to 1, got {sum(targets.values()):.4f}")
    if not 0 < float(cfg["rebalance"]["band_abs"]) < 1:
        raise ValueError("rebalance.band_abs must be between 0 and 1")
    if int(cfg["build_in"]["months"]) < 1:
        raise ValueError("build_in.months must be >= 1")
    if any(t >= 0 for t in cfg["drawdown_buffer"]["triggers"]):
        raise ValueError("drawdown triggers must be negative, e.g. -0.15")
    if cfg["contract"].get("leverage"):
        raise ValueError("leverage is not allowed by the contract")
    return cfg


def load_config(path: Path) -> Dict[str, Any]:
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def risk_targets(cfg: Dict[str, Any]) -> Dict[str, float]:
    return {s: w for s, w in cfg["targets"].items() if s != CASH}


__all__ = ["CASH", "is_usd", "validate", "load_config", "risk_targets"]
