"""Cash-flow simulator for the user's real trading rules (Snowball decision layer).

Rules simulated (all decisions use information available at signal date `t`; trades execute at the
next trading day's open, supplied as the price column):
- Fixed monthly contribution (default 4,500 TWD), credited on the first signal week of each month.
- Buy-only allocation: cash buys the top `n_buy` names whose score percentile >= `buy_pct`, in whole
  odd-lot shares, subject to `max_weight` per name. Taiwan fee = max(min_fee, fee_rate x value).
- Sells: a holding whose percentile falls below `sell_pct` is sold only if net proceeds exceed its
  cost basis (zero-realized-loss rule). `allow_loss_sells=True` lifts the rule for comparison.
  Sells pay fee + `sell_tax` (0.3% TW securities transaction tax).
- A held name with no price (delisted) is written down by `delist_haircut` and removed.

Outputs are TWD values; use `summarize` for money-weighted (XIRR) and time-weighted metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_DELIST_HAIRCUT = -0.30


@dataclass(frozen=True)
class CashflowConfig:
    monthly_cash: float = 4500.0
    n_buy: int = 1
    buy_pct: float = 0.98
    sell_pct: float = 0.60
    fee_rate: float = 0.001425
    min_fee: float = 1.0
    sell_tax: float = 0.003
    max_weight: float = 0.20
    allow_loss_sells: bool = False


@dataclass
class CashflowResult:
    values: pd.Series
    contributions: pd.Series
    cash: pd.Series
    final_positions: Dict[str, int]
    blocked_sells: int = 0
    realized_pnl: float = 0.0
    delist_events: int = 0
    n_trades: int = 0
    fees_paid: float = 0.0
    trade_log: List[dict] = field(default_factory=list)


def _fee(value: float, cfg: CashflowConfig) -> float:
    return max(cfg.min_fee, cfg.fee_rate * value) if value > 0 else 0.0


def _affordable_shares(budget: float, px: float, cfg: CashflowConfig) -> int:
    if px <= 0 or budget <= 0:
        return 0
    sh = int(np.floor(budget / (px * (1.0 + cfg.fee_rate))))
    while sh > 0 and sh * px + _fee(sh * px, cfg) > budget + 1e-9:
        sh -= 1
    return sh


class _Book:
    """Mutable-free bookkeeping helper: every update returns a new state dict copy."""

    def __init__(self) -> None:
        self.shares: Dict[str, int] = {}
        self.cost: Dict[str, float] = {}
        self.last_px: Dict[str, float] = {}


def _sell_decisions(
    book: _Book, today: pd.DataFrame, cfg: CashflowConfig, d: pd.Timestamp, res: CashflowResult,
) -> float:
    cash_in = 0.0
    for t in sorted(book.shares):
        if t not in today.index or not np.isfinite(today.at[t, "_pct"]):
            continue
        if today.at[t, "_pct"] >= cfg.sell_pct:
            continue
        px, sh = float(today.at[t, "_px"]), book.shares[t]
        gross = sh * px
        net = gross - _fee(gross, cfg) - cfg.sell_tax * gross
        if net <= book.cost[t] and not cfg.allow_loss_sells:
            res.blocked_sells += 1
            continue
        res.realized_pnl += net - book.cost[t]
        res.fees_paid += gross - net
        res.n_trades += 1
        res.trade_log.append({"date": d, "ticker": t, "side": "sell", "shares": sh, "px": px, "pnl": net - book.cost[t]})
        cash_in += net
        book.shares = {k: v for k, v in book.shares.items() if k != t}
        book.cost = {k: v for k, v in book.cost.items() if k != t}
    return cash_in


def _buy_decisions(
    book: _Book, today: pd.DataFrame, cfg: CashflowConfig, cash: float, d: pd.Timestamp, res: CashflowResult,
) -> float:
    equity = cash + sum(book.shares[t] * book.last_px.get(t, 0.0) for t in book.shares)
    cands = today.loc[today["_pct"] >= cfg.buy_pct].sort_values("_score", ascending=False)
    picks = []
    for t, row in cands.iterrows():
        held_val = book.shares.get(t, 0) * float(row["_px"])
        if equity > 0 and held_val / equity >= cfg.max_weight:
            continue
        picks.append(t)
        if len(picks) >= cfg.n_buy:
            break
    spent = 0.0
    for t in picks:
        budget = (cash - spent) / (len(picks) - picks.index(t))
        px = float(today.at[t, "_px"])
        cap = max(0.0, cfg.max_weight * equity - book.shares.get(t, 0) * px) if equity > 0 else budget
        sh = _affordable_shares(min(budget, cap) if cap > 0 else budget, px, cfg)
        if sh <= 0:
            continue
        outlay = sh * px + _fee(sh * px, cfg)
        spent += outlay
        res.fees_paid += outlay - sh * px
        res.n_trades += 1
        res.trade_log.append({"date": d, "ticker": t, "side": "buy", "shares": sh, "px": px, "pnl": 0.0})
        book.shares = {**book.shares, t: book.shares.get(t, 0) + sh}
        book.cost = {**book.cost, t: book.cost.get(t, 0.0) + outlay}
    return spent


def _fallback_px(fallback: Optional[pd.DataFrame], d: pd.Timestamp, t: str) -> float:
    if fallback is None or t not in fallback.columns or d not in fallback.index:
        return float("nan")
    return float(fallback.at[d, t])


def _handle_missing(
    book: _Book, today: pd.DataFrame, haircut: float, res: CashflowResult,
    fallback: Optional[pd.DataFrame], d: pd.Timestamp,
) -> float:
    """Held names absent from today's scored rows: keep them at the fallback price (e.g. they left the
    PIT universe but still trade); liquidate at last price x (1 + haircut) only when no price exists."""
    cash_in = 0.0
    for t in [t for t in book.shares if t not in today.index]:
        fb = _fallback_px(fallback, d, t)
        if np.isfinite(fb) and fb > 0:
            book.last_px = {**book.last_px, t: fb}
            continue
        value = book.shares[t] * book.last_px.get(t, 0.0) * (1.0 + haircut)
        res.realized_pnl += value - book.cost[t]
        res.delist_events += 1
        cash_in += value
        book.shares = {k: v for k, v in book.shares.items() if k != t}
        book.cost = {k: v for k, v in book.cost.items() if k != t}
    return cash_in


def simulate(
    df: pd.DataFrame, score_col: str, price_col: str, cfg: CashflowConfig,
    *, delist_haircut: float = DEFAULT_DELIST_HAIRCUT, price_fallback: Optional[pd.DataFrame] = None,
) -> CashflowResult:
    """Run the weekly cash-flow simulation on a tidy (date, ticker, score, price) frame.

    `price_fallback` (date x ticker, same units as `price_col`) values holdings that are missing from the
    scored rows, e.g. names that left a point-in-time universe but still trade.
    """
    data = df.dropna(subset=[score_col, price_col]).loc[lambda x: x[price_col] > 0].copy()
    data["_score"] = data[score_col].astype(float)
    data["_px"] = data[price_col].astype(float)
    data["_pct"] = data.groupby("date")["_score"].rank(pct=True)
    by_date = {d: g.set_index("ticker") for d, g in data.groupby("date")}
    dates = sorted(pd.to_datetime(data["date"].unique()))  # skip unsettled weeks with no prices at all

    book, cash, prev_month = _Book(), 0.0, None
    res = CashflowResult(values=pd.Series(dtype=float), contributions=pd.Series(dtype=float),
                         cash=pd.Series(dtype=float), final_positions={})
    vals, contribs, cashes = {}, {}, {}
    for d in dates:
        today = by_date.get(d, pd.DataFrame(columns=["_score", "_px", "_pct"]))
        month = (d.year, d.month)
        contribs[d] = cfg.monthly_cash if month != prev_month else 0.0
        cash += contribs[d]
        prev_month = month
        cash += _handle_missing(book, today, delist_haircut, res, price_fallback, d)
        book.last_px = {**book.last_px, **{t: float(today.at[t, "_px"]) for t in book.shares if t in today.index}}
        cash += _sell_decisions(book, today, cfg, d, res)
        cash -= _buy_decisions(book, today, cfg, cash, d, res)
        book.last_px = {**book.last_px, **today["_px"].astype(float).to_dict()}
        cashes[d] = cash
        vals[d] = cash + sum(sh * book.last_px.get(t, 0.0) for t, sh in book.shares.items())

    res.values, res.contributions, res.cash = pd.Series(vals), pd.Series(contribs), pd.Series(cashes)
    res.final_positions = dict(book.shares)
    return res


def dca_benchmark(
    px: pd.Series, monthly_cash: float, fee_rate: float = 0.001425, min_fee: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """Monthly DCA into one instrument (e.g. 0050) in whole shares; returns (values, contributions)."""
    cfg = CashflowConfig(monthly_cash=monthly_cash, fee_rate=fee_rate, min_fee=min_fee)
    shares, cash, prev, vals, contribs = 0, 0.0, None, {}, {}
    for d, p in px.dropna().items():
        month = (d.year, d.month)
        contribs[d] = monthly_cash if month != prev else 0.0
        cash += contribs[d]
        prev = month
        sh = _affordable_shares(cash, float(p), cfg)
        if sh > 0:
            cash -= sh * p + _fee(sh * p, cfg)
            shares += sh
        vals[d] = cash + shares * float(p)
    return pd.Series(vals), pd.Series(contribs)


def xirr(flows: pd.Series) -> float:
    """Annualised money-weighted return; flows negative = money in, final positive = value out."""
    t0 = flows.index.min()
    years = np.array([(d - t0).days / 365.25 for d in flows.index])
    amounts = flows.to_numpy(dtype=float)

    def npv(r: float) -> float:
        return float(np.sum(amounts / (1.0 + r) ** years))

    lo, hi = -0.99, 10.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def weekly_twr(values: pd.Series, contributions: pd.Series) -> pd.Series:
    """Weekly time-weighted returns with contribution flows removed: (V_t - C_t) / V_{t-1} - 1."""
    twr = (values - contributions) / values.shift(1) - 1.0
    return twr.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()


def summarize(values: pd.Series, contributions: pd.Series) -> Dict[str, float]:
    """Money-weighted XIRR plus time-weighted CAGR / MDD that strip out the contribution flows."""
    flows = -contributions.copy()
    flows.iloc[-1] += values.iloc[-1]
    twr = weekly_twr(values, contributions)
    curve = (1.0 + twr).cumprod()
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1e-9)
    return {
        "Final_Value": float(values.iloc[-1]),
        "Contributed": float(contributions.sum()),
        "Profit": float(values.iloc[-1] - contributions.sum()),
        "XIRR": xirr(flows),
        "TWR_CAGR": float(curve.iloc[-1] ** (1.0 / years) - 1.0) if len(curve) else float("nan"),
        "TWR_MDD": float((curve / curve.cummax() - 1.0).min()) if len(curve) else float("nan"),
    }
