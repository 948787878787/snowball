"""Suggestion rules for the core ETF account. They only print suggestions; they never place orders.

  build_in        : the risk targets are bought in `build_in.months` equal monthly tranches.
  rebalance       : every `rebalance.every_months` months after build-in, trade back to target only the
                    lines whose weight is more than `band_abs` away from target.
  drawdown_deploy : when the index is `trigger` below its peak, release the cash buffer in equal parts
                    (one part per trigger reached), split across the risk targets by target weight.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import pandas as pd

from etf_config import CASH, is_usd, risk_targets

USD_SHARE_DECIMALS = 3


def months_between(start: pd.Timestamp, day: pd.Timestamp) -> int:
    return (day.year - start.year) * 12 + (day.month - start.month)


def shares_for(symbol: str, amount_twd: float, price_twd: float) -> Union[int, float]:
    """Whole odd-lot shares for TW listings; fractional shares (3 decimals) for USD ETFs bought via
    sub-brokerage, so a tranche smaller than one share is not rounded down to zero."""
    if price_twd <= 0:
        return 0
    if is_usd(symbol):
        return round(amount_twd / price_twd, USD_SHARE_DECIMALS)
    return int(amount_twd // price_twd)


def format_shares(shares: Union[int, float]) -> str:
    return str(shares) if isinstance(shares, int) else f"{shares:.{USD_SHARE_DECIMALS}f}"


def build_in(cfg: Dict[str, Any], start: pd.Timestamp, as_of: pd.Timestamp, invested: Dict[str, float],
             price_twd: Dict[str, float]) -> List[Dict[str, Any]]:
    months = int(cfg["build_in"]["months"])
    k = months_between(start, as_of)
    if k < 0 or k >= months:
        return []
    capital = float(cfg["initial_capital_twd"])
    out = []
    for sym, w in risk_targets(cfg).items():
        amount = max(0.0, capital * w * (k + 1) / months - invested.get(sym, 0.0))
        if amount > 0:
            out.append({"symbol": sym, "amount_twd": amount, "shares": shares_for(sym, amount, price_twd[sym]),
                        "reason": f"建倉第 {k + 1}/{months} 批"})
    return out


def rebalance(cfg: Dict[str, Any], start: pd.Timestamp, as_of: pd.Timestamp, weights: Dict[str, float],
              nav: float) -> Dict[str, Any]:
    every, band = int(cfg["rebalance"]["every_months"]), float(cfg["rebalance"]["band_abs"])
    m = months_between(start, as_of)
    due = m >= int(cfg["build_in"]["months"]) and m > 0 and m % every == 0
    if not due:
        return {"due": False, "trades": []}
    trades = []
    for sym, target in risk_targets(cfg).items():
        gap = target - weights.get(sym, 0.0)
        if abs(gap) > band + 1e-12:
            trades.append({"symbol": sym, "amount_twd": gap * nav, "reason": f"偏離目標 {-gap:+.1%}"})
    return {"due": True, "trades": trades}


def drawdown_deploy(cfg: Dict[str, Any], index_close: pd.Series, cash: float, nav: float) -> Dict[str, Any]:
    dd_cfg = cfg["drawdown_buffer"]
    s = index_close.dropna().iloc[-int(dd_cfg["lookback_days"]):]
    drawdown = float(s.iloc[-1] / s.max() - 1.0) if len(s) else 0.0
    triggers = sorted(dd_cfg["triggers"], reverse=True)
    hit = sum(1 for t in triggers if drawdown <= t)
    buffer = float(cfg["targets"][CASH]) * nav
    allowed = buffer * (1.0 - hit / len(triggers))
    deploy = max(0.0, cash - allowed) if hit else 0.0
    risk = risk_targets(cfg)
    total = sum(risk.values())
    return {"drawdown": drawdown, "triggered": hit, "deploy_twd": deploy,
            "split": {sym: deploy * w / total for sym, w in risk.items()}}


__all__ = ["months_between", "shares_for", "format_shares", "build_in", "rebalance", "drawdown_deploy"]
