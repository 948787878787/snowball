"""Snowball v8 MLOps Model Decay Monitor, Sloan Accruals Quality & Core-Satellite Allocator.

References:
- Sloan (1996), *The Accounting Review*: Accruals quality (`-(NI - OCF) / TotalAssets`) detects low-cash accounting earnings.
- Moreira & Muir (2017), *Journal of Finance*: Ex-ante volatility-managed allocation to defensive cash/T-Bills.
- Gilchrist & Zakrajsek (2012), *American Economic Review*: Credit-spread macro stress conditioning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def compute_accruals_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Sloan (1996) cash-backed earnings quality `-(net_income - ocf) / total_assets`."""
    out = df.copy()
    if not {"net_income", "ocf", "total_assets"}.issubset(out.columns):
        out["accruals_quality"] = np.nan
        return out
    assets = out["total_assets"].astype(float).replace(0.0, np.nan)
    accrual = (out["net_income"].astype(float) - out["ocf"].astype(float)) / assets
    out["accruals_quality"] = (-accrual).clip(lower=-0.50, upper=0.50)
    return out


def allocate_core_satellite(
    *,
    macro_stress: float = 0.0,
    realized_tw_vol: float = 0.15,
    base_us_beta: float = 0.50,
    base_tw_alpha: float = 0.40,
    base_cash: float = 0.10,
    vol_target: float = 0.15,
) -> Dict[str, float]:
    """Allocate capital across US Beta ETF, TW v7 Alpha Satellite, and Defensive Cash/T-Bills."""
    vol_scale = min(1.0, float(vol_target) / max(float(realized_tw_vol), 1e-6))
    stress_damp = max(0.50, 1.0 - 0.15 * max(float(macro_stress), 0.0))
    tw_w = round(float(base_tw_alpha) * vol_scale * stress_damp, 4)
    us_w = round(float(base_us_beta) * max(0.75, stress_damp), 4)
    cash_w = round(max(float(base_cash), 1.0 - tw_w - us_w), 4)
    total = us_w + tw_w + cash_w
    return {
        "us_beta_etf": round(us_w / total, 4),
        "tw_alpha_satellite": round(tw_w / total, 4),
        "cash_tbill": round(1.0 - round(us_w / total, 4) - round(tw_w / total, 4), 4),
    }


def evaluate_model_health(
    ic_series: pd.Series,
    latest_signals: pd.DataFrame,
    *,
    macro_stress: float = 0.0,
    max_industry_cap: float = 0.25,
) -> Dict[str, object]:
    """Evaluate rolling 12w/26w Rank IC decay, industry concentration, and circuit-breaker exposure scale."""
    ic_clean = ic_series.dropna().astype(float)
    ic_12w = float(ic_clean.tail(12).mean()) if len(ic_clean) >= 4 else np.nan
    ic_26w = float(ic_clean.tail(26).mean()) if len(ic_clean) >= 8 else np.nan

    max_ind_w = 0.0
    if latest_signals is not None and not latest_signals.empty and "industry" in latest_signals.columns:
        max_ind_w = float(latest_signals["industry"].value_counts(normalize=True).max())

    decay_flag = np.isfinite(ic_26w) and ic_26w < 0.0
    ind_breach = max_ind_w > float(max_industry_cap) + 1e-6
    high_stress = float(macro_stress) >= 1.5

    if decay_flag and (high_stress or ind_breach):
        status, scale = "DECAY_HALT", 0.50
    elif decay_flag or ind_breach or high_stress:
        status, scale = "CAUTION", 0.75
    else:
        status, scale = "HEALTHY", 1.00

    return {
        "status": status,
        "rolling_12w_ic": round(ic_12w, 5) if np.isfinite(ic_12w) else None,
        "rolling_26w_ic": round(ic_26w, 5) if np.isfinite(ic_26w) else None,
        "max_industry_weight": round(max_ind_w, 4),
        "macro_stress": round(float(macro_stress), 3),
        "recommended_exposure_scale": scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball v8 MLOps Model Health & Core-Satellite Report")
    parser.add_argument("--oos-parquet", type=Path, required=True, help="OOS predictions parquet")
    parser.add_argument("--signals-csv", type=Path, required=True, help="Latest production signals CSV")
    parser.add_argument("--score-col", default="pred_v7_final", help="Prediction column name")
    parser.add_argument("--macro-stress", type=float, default=0.0, help="Latest lagged macro_stress Z-score")
    args = parser.parse_args()

    import train_walkforward as twf

    oos = pd.read_parquet(args.oos_parquet)
    t_col = "target_rank_4w" if "target_rank_4w" in oos.columns else "target_rank_mkt_4w"
    ic_s = twf._compute_daily_ic_series(oos, args.score_col, t_col, market_filter="TW")
    sig = pd.read_csv(args.signals_csv)
    health = evaluate_model_health(ic_s, sig, macro_stress=args.macro_stress)
    alloc = allocate_core_satellite(macro_stress=args.macro_stress)
    print(json.dumps({"model_health": health, "core_satellite_allocation": alloc}, indent=2))


if __name__ == "__main__":
    main()
