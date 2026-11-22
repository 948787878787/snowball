"""Daily simulator for the ETF risk-control research (direction A, pre-registered in etf_risk_v1).

Sleeves: `tw`, `us` (TWD total-return price levels on the TW trading calendar) and `cash`.
Rules decide at month-end closes using data up to and including that date; trades execute at the
next trading day's close. Costs: buys pay fee + slippage, sells pay fee + ETF tax + slippage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SLEEVES = ("tw", "us", "cash")
TRADING_DAYS = 252
Weights = Dict[str, float]


def align_strictly_before(series: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Value of `series` from the last date strictly before each day (US close is after the TW close)."""
    s = series.dropna().sort_index()
    pos = s.index.searchsorted(days, side="left") - 1
    vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=days, dtype=float)


def month_end_days(days: pd.DatetimeIndex, start) -> List[pd.Timestamp]:
    ends = pd.Series(days, index=days).groupby([days.year, days.month]).max()
    return [d for d in ends if d >= pd.Timestamp(start)]


def next_day(days: pd.DatetimeIndex, day: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = days.searchsorted(day, side="right")
    return days[pos] if pos < len(days) else None


def _banded(target: Weights, current: Weights, band: float, force: frozenset = frozenset()) -> Optional[Weights]:
    """Move each risk sleeve that is off by more than `band` (or forced) to target; cash absorbs."""
    moved = {s: target[s] for s in ("tw", "us") if s in force or abs(current[s] - target[s]) > band + 1e-12}
    if not moved:
        return None
    new = {**{s: current[s] for s in ("tw", "us")}, **moved}
    return {**new, "cash": 1.0 - new["tw"] - new["us"]}


@dataclass(frozen=True)
class StaticRule:
    targets: Weights
    band: float
    months: tuple = (6, 12)

    def decide(self, day: pd.Timestamp, current: Weights, prices: Optional[pd.DataFrame]) -> Optional[Weights]:
        return _banded(self.targets, current, self.band) if day.month in self.months else None

    def initial(self, day, prices) -> Weights:
        return dict(self.targets)


@dataclass(frozen=True)
class TrendRule:
    targets: Weights
    sma_days: int
    band: float

    def rule_weights(self, day: pd.Timestamp, prices: pd.DataFrame) -> Weights:
        hist = prices.loc[:day]
        w = {s: self.targets[s] if hist[s].iloc[-1] >= hist[s].iloc[-self.sma_days:].mean() else 0.0 for s in ("tw", "us")}
        return {**w, "cash": 1.0 - w["tw"] - w["us"]}

    def decide(self, day, current, prices) -> Optional[Weights]:
        target = self.rule_weights(day, prices)
        flipped = frozenset(s for s in ("tw", "us") if (current[s] > 0) != (target[s] > 0))
        return _banded(target, current, self.band, flipped)

    def initial(self, day, prices) -> Weights:
        return self.rule_weights(day, prices)


@dataclass(frozen=True)
class VolTargetRule:
    targets: Weights
    vol_target: float
    window: int
    band: float

    def rule_weights(self, day: pd.Timestamp, prices: pd.DataFrame) -> Weights:
        rets = prices.loc[:day, ["tw", "us"]].pct_change().iloc[-self.window:]
        risk = self.targets["tw"] + self.targets["us"]
        mix = rets["tw"] * self.targets["tw"] / risk + rets["us"] * self.targets["us"] / risk
        sigma = float(mix.std() * np.sqrt(TRADING_DAYS))
        total = min(risk, self.vol_target / sigma) if sigma > 0 else risk
        return {"tw": total * self.targets["tw"] / risk, "us": total * self.targets["us"] / risk, "cash": 1.0 - total}

    def decide(self, day, current, prices) -> Optional[Weights]:
        return _banded(self.rule_weights(day, prices), current, self.band)

    def initial(self, day, prices) -> Weights:
        return self.rule_weights(day, prices)


@dataclass(frozen=True)
class HoldRule:
    targets: Weights

    def decide(self, day, current, prices) -> Optional[Weights]:
        return None

    def initial(self, day, prices) -> Weights:
        return dict(self.targets)


@dataclass
class SimResult:
    nav: pd.Series
    weights: pd.DataFrame
    costs_paid: float = 0.0
    traded_value: float = 0.0
    n_trade_days: int = 0
    trades: List[dict] = field(default_factory=list)


def _trade(values: Dict[str, float], target: Weights, costs: Dict[str, float]) -> tuple:
    nav = sum(values.values())
    delta = {s: target[s] * nav - values[s] for s in ("tw", "us")}
    buy = sum(v for v in delta.values() if v > 0)
    sell = -sum(v for v in delta.values() if v < 0)
    cost = buy * (costs["fee_rate"] + costs["slippage_each_side"]) + \
        sell * (costs["fee_rate"] + costs["sell_tax_etf"] + costs["slippage_each_side"])
    new = {"tw": values["tw"] + delta["tw"], "us": values["us"] + delta["us"]}
    new["cash"] = nav - new["tw"] - new["us"] - cost
    return new, cost, buy + sell


def simulate(prices: pd.DataFrame, rule, signal_start, cash_yield: float, costs: Dict[str, float],
             initial_nav: float = 100000.0) -> SimResult:
    days = prices.index
    signals = month_end_days(days, signal_start)
    execs = {next_day(days, d): d for d in signals if next_day(days, d) is not None}
    first = min(execs)
    growth = (1.0 + cash_yield) ** (1.0 / TRADING_DAYS)
    values = {"tw": 0.0, "us": 0.0, "cash": float(initial_nav)}
    res = SimResult(nav=pd.Series(dtype=float), weights=pd.DataFrame())
    navs, wts = {}, {}
    for i, d in enumerate(days[days.get_loc(first):], start=days.get_loc(first)):
        if d != first:
            prev = days[i - 1]
            values = {"tw": values["tw"] * prices.at[d, "tw"] / prices.at[prev, "tw"],
                      "us": values["us"] * prices.at[d, "us"] / prices.at[prev, "us"],
                      "cash": values["cash"] * growth}
        if d in execs:
            nav = sum(values.values())
            current = {s: values[s] / nav for s in SLEEVES}
            target = rule.initial(execs[d], prices) if d == first else rule.decide(execs[d], current, prices)
            if target is not None:
                values, cost, traded = _trade(values, target, costs)
                res.costs_paid += cost
                res.traded_value += traded
                res.n_trade_days += 1
                res.trades.append({"date": d, "signal": execs[d], **{f"w_{s}": target[s] for s in SLEEVES}})
        nav = sum(values.values())
        navs[d] = nav
        wts[d] = {s: values[s] / nav for s in SLEEVES}
    res.nav = pd.Series(navs)
    res.weights = pd.DataFrame.from_dict(wts, orient="index")
    return res


__all__ = ["align_strictly_before", "month_end_days", "next_day", "StaticRule", "TrendRule", "VolTargetRule",
           "HoldRule", "SimResult", "simulate"]
