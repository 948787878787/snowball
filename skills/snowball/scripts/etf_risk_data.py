"""Data preparation for the ETF risk-control research (direction A). No strategy returns here.

Sleeves (TWD, total return):
  TW : 006208.TW adjusted close (Taiwan 50; clean on Yahoo since 2012-06-22). 0050.TW is NOT used:
       its Yahoo history has a spurious >15% one-day move on 2014-01-02.
  US : SPY adjusted close (USD) x TWD=X, minus an annual fee drag approximating 00646.TW; TWD=X
       one-day spikes of more than 20% that reverse are removed first (bad ticks 1.80 on 2011-10-25 and 3.67 on 2014-12-31; genuine moves such as May 2025 are kept).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def clean_reversing_spikes(series: pd.Series, threshold: float = 0.20) -> pd.Series:
    """Replace a value that jumps more than `threshold` from the previous value and then jumps back
    (next value within `threshold` of the previous one) with the previous value."""
    s = series.astype(float)
    prev, nxt = s.shift(1), s.shift(-1)
    spike = ((s / prev - 1).abs() > threshold) & ((nxt / prev - 1).abs() <= threshold)
    return s.where(~spike, prev)


def us_sleeve_twd(usd_total_return: pd.Series, fx_twd_per_usd: pd.Series, annual_fee: float) -> pd.Series:
    idx = usd_total_return.index.intersection(fx_twd_per_usd.index)
    level = usd_total_return.reindex(idx) * fx_twd_per_usd.reindex(idx)
    drag = (1.0 - annual_fee) ** (np.arange(len(idx)) / TRADING_DAYS)
    return level * drag


def tracking_difference(proxy: pd.Series, actual: pd.Series) -> float:
    """Annualised log return of `proxy` minus `actual` over their common dates."""
    idx = proxy.dropna().index.intersection(actual.dropna().index)
    years = len(idx) / TRADING_DAYS
    gap = np.log(proxy[idx].iloc[-1] / proxy[idx].iloc[0]) - np.log(actual[idx].iloc[-1] / actual[idx].iloc[0])
    return float(gap / years)


__all__ = ["clean_reversing_spikes", "us_sleeve_twd", "tracking_difference"]
