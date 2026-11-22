"""Structure B profit split: non-cumulative annual hurdle with a high-water mark, no clawback.

Per contract year (anniversaries of `start`): S = NAV before the year, C = net external money in during
the year (deposits - withdrawals; carry payments excluded), E = NAV at year end (or `as_of` when the
year is still running, a provisional estimate using the full-year hurdle).
  threshold = max(HWM + C, (S + C) * (1 + hurdle))
  carry     = max(0, E - threshold) * manager_share
  HWM_next  = max(HWM + C, E - carry)
Assumes each year's carry is paid at the anniversary and recorded in the ledger as a `carry` row.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _nav_at(nav: pd.Series, day: pd.Timestamp) -> float:
    s = nav.loc[:day]
    return float(s.iloc[-1]) if len(s) else 0.0


def contract_years(start: pd.Timestamp, as_of: pd.Timestamp) -> List[tuple]:
    years, y0 = [], pd.Timestamp(start)
    while y0 <= as_of:
        y1 = y0 + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        years.append((y0, min(y1, as_of), y1 <= as_of))
        y0 = y1 + pd.Timedelta(days=1)
    return years


def contract_split(nav: pd.Series, flows: pd.Series, start: pd.Timestamp, as_of: pd.Timestamp,
                   hurdle: float = 0.06, manager_share: float = 1.0) -> List[Dict[str, Any]]:
    hwm, out = 0.0, []
    for y0, y1, complete in contract_years(start, as_of):
        s_nav = _nav_at(nav, y0 - pd.Timedelta(days=1))
        contrib = float(flows.loc[(flows.index >= y0) & (flows.index <= y1)].sum())
        end = _nav_at(nav, y1)
        threshold = max(hwm + contrib, (s_nav + contrib) * (1.0 + hurdle))
        carry = max(0.0, end - threshold) * manager_share
        out.append({"year_start": y0, "year_end": y1, "complete": complete, "start_nav": s_nav,
                    "contributions": contrib, "end_nav": end, "threshold": threshold, "carry": carry,
                    "investor_gain": end - carry - s_nav - contrib,
                    "year_return": (end / (s_nav + contrib) - 1.0) if s_nav + contrib > 0 else 0.0,
                    "hwm_after": max(hwm + contrib, end - carry)})
        hwm = out[-1]["hwm_after"]
    return out


__all__ = ["contract_years", "contract_split"]
