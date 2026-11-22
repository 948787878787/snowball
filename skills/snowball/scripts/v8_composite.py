"""Snowball v8 pre-registered zero-parameter composite and frozen-config utilities.

Every sign and weight comes from `configs/v8_frozen.json`; nothing here is estimated from returns.
  dim_d   = mean over the dimension's inputs of sign * cross-sectional z (winsorised)
  v8_zero = sum_d w_d * dim_d / sum of w_d over dimensions available for that row
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_hash(cfg: Dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON (sorted keys), used to stamp every paper-trading record."""
    canon = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def cs_zscore(df: pd.DataFrame, col: str, winsor: float, min_names: int, by: str = "date",
              winsor_pct: float = 0.01) -> pd.Series:
    """Per-date robust z-score: raw values clipped to the date's [winsor_pct, 1 - winsor_pct] quantiles
    (so a few extreme ratios cannot inflate the std), standardised, then clipped to +/- winsor.
    NaN when fewer than `min_names` values that date."""
    grp = df.groupby(by)[col]
    lo, hi = grp.transform(lambda s: s.quantile(winsor_pct)), grp.transform(lambda s: s.quantile(1.0 - winsor_pct))
    clipped = df[col].clip(lower=lo, upper=hi)
    cgrp = clipped.groupby(df[by])
    z = ((clipped - cgrp.transform("mean")) / cgrp.transform("std")).clip(-winsor, winsor)
    enough = grp.transform("count") >= min_names
    return z.where(enough)


def add_v8_zero(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add one `dim_<name>` column per dimension and the weighted `v8_zero` composite."""
    zc = cfg["zscore"]
    needed = [c for dim in cfg["dimensions"].values() for c in dim]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"v8 composite inputs missing from panel: {missing}")
    out = df.copy()
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for name, inputs in cfg["dimensions"].items():
        parts = pd.concat([sign * cs_zscore(df, col, zc["winsor"], zc["min_names"], zc["by"], zc.get("winsor_pct", 0.01))
                           for col, sign in inputs.items()], axis=1)
        dim = parts.mean(axis=1, skipna=True)
        out[f"dim_{name}"] = dim
        w = float(cfg["dimension_weights"][name])
        num = num + (w * dim).fillna(0.0)
        den = den + w * dim.notna()
    out["v8_zero"] = (num / den).where(den > 0)
    return out


def industry_demean(df: pd.DataFrame, col: str, industry_col: str = "industry") -> pd.Series:
    return df[col] - df.groupby(["date", industry_col])[col].transform("mean")


def rank_pct(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True)


def blend_ranks(df: pd.DataFrame, cols: Dict[str, float]) -> pd.Series:
    """Weighted average of per-date percentile ranks; renormalised over available inputs."""
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for col, w in cols.items():
        r = rank_pct(df, col)
        num = num + (w * r).fillna(0.0)
        den = den + w * r.notna()
    return (num / den).where(den > 0).astype(float)


__all__ = ["load_config", "config_hash", "cs_zscore", "add_v8_zero", "industry_demean", "rank_pct", "blend_ranks"]
