"""Metrics and the pre-registered gate for the ETF risk-control research (etf_risk_v1)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import evaluate as ev
from v8_eval_metrics import rolling_hurdle_windows, time_under_water

MDD_IMPROVEMENT = 0.05
CAGR_SHORTFALL = 0.015


def performance(nav: pd.Series) -> Dict[str, float]:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    mdd = float((nav / nav.cummax() - 1.0).min())
    tuw = time_under_water(nav.resample("W-FRI").last().dropna())
    return {"cagr": cagr, "mdd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "vol": float(nav.pct_change().std() * np.sqrt(252)),
            "max_weeks_under_water": tuw["max_weeks_under_water"],
            "current_weeks_under_water": tuw["current_weeks_under_water"]}


def calendar_returns(nav: pd.Series) -> pd.Series:
    """Returns of full calendar years only (first and last partial years dropped)."""
    year_end = nav.groupby(nav.index.year).last()
    rets = year_end.pct_change().dropna()
    last_full = nav.index[-1].year - (0 if nav.index[-1].month == 12 and nav.index[-1].day >= 28 else 1)
    return rets.loc[rets.index <= last_full]


def monthly_excess(nav: pd.Series, base: pd.Series) -> Dict[str, float]:
    m = nav.resample("ME").last().pct_change().dropna()
    b = base.resample("ME").last().pct_change().dropna()
    ex = (m - b).dropna()
    return {"mean_monthly": float(ex.mean()), "t_nw": float(ev.newey_west_t(ex, lags=4))}


def gate(arm: Dict[str, Any], s0: Dict[str, Any]) -> Dict[str, bool]:
    f, b = arm["full"], s0["full"]
    out = {
        "mdd_5pp_shallower": bool(abs(f["mdd"]) <= abs(b["mdd"]) - MDD_IMPROVEMENT + 1e-12),
        "calmar_higher_full_and_halves": bool(f["calmar"] > b["calmar"] and arm["h1"]["calmar"] > s0["h1"]["calmar"]
                                              and arm["h2"]["calmar"] > s0["h2"]["calmar"]),
        "cagr_within_1p5pp": bool(f["cagr"] >= b["cagr"] - CAGR_SHORTFALL - 1e-12),
        "tuw_not_longer": bool(f["max_weeks_under_water"] <= b["max_weeks_under_water"]),
    }
    return {**out, "pass": all(out.values())}


def pick_forward(results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    passers = {k: v for k, v in results.items() if v["gate"]["pass"]}
    return max(passers, key=lambda k: passers[k]["full"]["calmar"]) if passers else None


def describe(nav: pd.Series, halves) -> Dict[str, Any]:
    weekly = nav.resample("W-FRI").last().pct_change().dropna()
    years = calendar_returns(nav)
    return {
        "full": performance(nav),
        "h1": performance(nav.loc[halves[0][0]:halves[0][1]]),
        "h2": performance(nav.loc[halves[1][0]:halves[1][1]]),
        "calendar_years": {int(y): float(r) for y, r in years.items()},
        "full_years_below_6pct": int((years < 0.06).sum()), "n_full_years": int(len(years)),
        "hurdle_3y": rolling_hurdle_windows(weekly, hurdle=0.06, years=3),
    }


__all__ = ["performance", "calendar_returns", "monthly_excess", "gate", "pick_forward", "describe"]
