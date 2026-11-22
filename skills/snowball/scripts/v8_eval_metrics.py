"""Investor-facing metrics for the v8 evaluation (structure B: 6% non-cumulative hurdle with a high-water mark).

Contract modelled (per year, starting NAV S, high-water mark H, ending NAV E before carry):
  threshold = max(H, S * (1 + hurdle))
  carry     = max(0, E - threshold)          # everything above the threshold goes to the manager
  investor NAV after carry = E - carry;  H_next = max(H, investor NAV)
Losses stay with the investor (no clawback), the hurdle does not accumulate across years, no leverage.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


def time_under_water(nav: pd.Series) -> Dict[str, object]:
    """Longest and current run of consecutive weeks below the running peak."""
    below = (nav < nav.cummax() - 1e-12).to_numpy()
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return {"max_weeks_under_water": int(longest), "current_weeks_under_water": int(current),
            "recovered": bool(current == 0)}


def hwm_hurdle_path(yearly_returns: Sequence[float], hurdle: float = 0.06, start: float = 100.0) -> Dict[str, List[float]]:
    nav, hwm = float(start), float(start)
    carry, inv = [], []
    for r in yearly_returns:
        end = nav * (1.0 + float(r))
        c = max(0.0, end - max(hwm, nav * (1.0 + hurdle)))
        nav = end - c
        hwm = max(hwm, nav)
        carry.append(c)
        inv.append(nav)
    return {"carry": carry, "investor_nav": inv}


def rolling_hurdle_windows(weekly_returns: pd.Series, hurdle: float = 0.06, years: int = 3) -> Dict[str, float]:
    """Every rolling `years`-year window of weekly net returns, split into 52-week contract years.
    Overlapping windows are highly dependent: descriptive only."""
    r = weekly_returns.dropna().to_numpy(dtype=float)
    span = WEEKS_PER_YEAR * years
    n = len(r) - span + 1
    if n <= 0:
        return {"n_windows": 0, "p_any_carry": np.nan, "p_investor_hit_hurdle_every_year": np.nan,
                "mean_manager_share_of_profit": np.nan}
    any_carry, every_year, shares = [], [], []
    for s in range(n):
        block = r[s:s + span].reshape(years, WEEKS_PER_YEAR)
        yearly = np.prod(1.0 + block, axis=1) - 1.0
        path = hwm_hurdle_path(yearly, hurdle)
        total_profit = 100.0 * (np.prod(1.0 + yearly) - 1.0)
        carry = float(np.sum(path["carry"]))
        any_carry.append(carry > 0)
        every_year.append(bool(np.all(yearly >= hurdle)))
        shares.append(carry / total_profit if total_profit > 0 else 0.0)
    return {"n_windows": int(n), "p_any_carry": float(np.mean(any_carry)),
            "p_investor_hit_hurdle_every_year": float(np.mean(every_year)),
            "mean_manager_share_of_profit": float(np.mean(shares))}


__all__ = ["time_under_water", "hwm_hurdle_path", "rolling_hurdle_windows"]
