"""NAV, time-weighted returns and drawdown metrics for the core ETF account (TWD).

Prices are native closes (TWD for .TW/.TWO, USD otherwise) with dates x symbols; `fx` is TWD per USD.
External flows are treated as arriving at the start of their day:
  r_t = NAV_t / (NAV_{t-1} + flow_t) - 1
so deposits and withdrawals never show up as performance.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

import etf_ledger as el
from etf_config import is_usd
from v8_eval_metrics import time_under_water


def holdings_value(led: pd.DataFrame, prices: pd.DataFrame, fx: pd.Series, day: pd.Timestamp) -> Dict[str, float]:
    out = {}
    for sym, sh in el.positions(led, day).items():
        px = float(prices[sym].loc[:day].dropna().iloc[-1])
        out[sym] = sh * px * (float(fx.loc[:day].dropna().iloc[-1]) if is_usd(sym) else 1.0)
    return out


def nav_series(led: pd.DataFrame, prices: pd.DataFrame, fx: pd.Series, as_of) -> pd.Series:
    days = pd.bdate_range(led["date"].min(), pd.Timestamp(as_of))
    px = prices.reindex(prices.index.union(days)).sort_index().ffill().reindex(days)
    rate = fx.reindex(fx.index.union(days)).sort_index().ffill().reindex(days)
    vals = []
    for d in days:
        pos = el.positions(led, d)
        stock = sum(sh * px.at[d, s] * (rate.at[d] if is_usd(s) else 1.0) for s, sh in pos.items())
        vals.append(el.cash_balance(led, d) + stock)
    return pd.Series(vals, index=days, dtype=float)


def twr_index(nav: pd.Series, flows: pd.Series) -> pd.Series:
    f = flows.reindex(nav.index, fill_value=0.0)
    base = nav.shift(1).fillna(0.0) + f
    r = (nav / base.where(base > 0) - 1.0).fillna(0.0)
    return (1.0 + r).cumprod()


def period_return(index: pd.Series, start, end) -> float:
    """Return over [start, end] using the last index value before `start` as the base."""
    before = index.loc[: pd.Timestamp(start) - pd.Timedelta(days=1)]
    base = float(before.iloc[-1]) if len(before) else 1.0
    inside = index.loc[: pd.Timestamp(end)]
    return float(inside.iloc[-1] / base - 1.0) if len(inside) else np.nan


def benchmark_return(adj_close: pd.Series, start, end) -> float:
    before = adj_close.loc[: pd.Timestamp(start) - pd.Timedelta(days=1)].dropna()
    inside = adj_close.loc[: pd.Timestamp(end)].dropna()
    if before.empty or inside.empty:
        return np.nan
    return float(inside.iloc[-1] / before.iloc[-1] - 1.0)


def risk_metrics(index: pd.Series) -> Dict[str, object]:
    weekly = index.resample("W-FRI").last().dropna()
    tuw = time_under_water(weekly)
    return {"mdd": float((index / index.cummax() - 1.0).min()),
            "max_weeks_under_water": tuw["max_weeks_under_water"],
            "current_weeks_under_water": tuw["current_weeks_under_water"]}


__all__ = ["holdings_value", "nav_series", "twr_index", "period_return", "benchmark_return", "risk_metrics"]
