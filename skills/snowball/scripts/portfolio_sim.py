"""Realistic Multi-Market Portfolio TWR Simulator & Trial Logger (Snowball v7 Stage 1).

Implements Stage 1.3 & Principle #2 requirements:
- Market-specific real round-trip costs:
  * TW board lot : 0.585% (0.1425% x 2 commission + 0.30% securities transaction tax)
  * TW odd lot   : 1.185% (0.585% + 0.30% x 2 odd-lot liquidity slippage)
  * US           : 0.15% round-trip + 30% dividend withholding drag (0.30 x ~1.5% = 0.45%/yr)
- Conservative delisting handling: if a held stock disappears with NaN forward return, applies
  `delist_return = -0.30` once and removes it from `held`.
- Limit-up / limit-down lock handling (TW +/-10%): skips buying names locked limit-up and defers
  selling names locked limit-down when `is_limit_up` / `is_limit_down` columns are present.
- Reports true per-rebalance turnover (`Avg_Rebalance_Turnover`), `Annual_Round_Trips`, and
  `Avg_Weekly_Turnover`.
- Appends every experimental run to `results/trial_log.csv` for Deflated Sharpe accounting.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

TW_BOARD_LOT_COST = 0.00585
TW_ODD_LOT_COST = 0.01185
US_ROUND_TRIP_COST = 0.0015
DEFAULT_US_DIV_WITHHOLD_ANNUAL = 0.0045  # 30% withholding on ~1.5% US dividend yield
DEFAULT_DELIST_RETURN = -0.30


def _stock_round_trip_cost(market: str, tw_odd_lot: bool) -> float:
    if str(market).upper() == "TW":
        return TW_ODD_LOT_COST if tw_odd_lot else TW_BOARD_LOT_COST
    return US_ROUND_TRIP_COST


def _apply_membership_mask(df: pd.DataFrame, membership_mask: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Filter tidy panel `(date, ticker)` rows against a point-in-time boolean `membership_mask`."""
    if membership_mask is None or membership_mask.empty:
        return df
    mask = membership_mask.copy()
    mask.index = pd.to_datetime(mask.index).astype("datetime64[ns]")
    long_m = mask.stack().rename("_is_member").reset_index()
    long_m.columns = ["date", "ticker", "_is_member"]
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    joined = out.merge(long_m, on=["date", "ticker"], how="left")
    keep = joined["_is_member"].fillna(False).astype(bool)
    return joined.loc[keep].drop(columns=["_is_member"]).reset_index(drop=True)


def _select_cohort_target(grp: pd.DataFrame, score_col: str, top_k: int) -> Set[str]:
    ranked = grp.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    if "is_limit_up" in ranked.columns:
        ranked = ranked[~ranked["is_limit_up"].fillna(False).astype(bool)]
    k = min(top_k, max(4, len(ranked) // 10))
    return set(ranked["ticker"].iloc[:k].tolist())


def _compute_rebalance_cost(
    new_set: Set[str],
    held: Set[str],
    mkt_map: Dict[str, str],
    default_market: str,
    tw_odd_lot: bool,
    override_cost: Optional[float],
) -> tuple[float, float]:
    """Return `(portfolio_cost_fraction, rebalance_turnover_fraction)`."""
    if not held:
        return 0.0, 0.0
    entered = new_set - held
    exited = held - new_set
    n_slots = max(len(new_set), 1)
    turn = len(entered) / n_slots
    if override_cost is not None:
        return turn * override_cost, turn
    # One-way buy cost for entries + one-way sell cost for exits (half of round-trip each side)
    buy_c = sum(0.5 * _stock_round_trip_cost(mkt_map.get(t, default_market), tw_odd_lot) for t in entered) / n_slots
    sell_c = sum(0.5 * _stock_round_trip_cost(mkt_map.get(t, default_market), tw_odd_lot) for t in exited) / max(len(held), 1)
    return float(buy_c + sell_c), float(turn)


def _run_single_cohort(
    offset: int,
    dates: List[pd.Timestamp],
    by_date: Dict[pd.Timestamp, pd.DataFrame],
    score_col: str,
    top_k: int,
    rebalance_every: int,
    mkt_map: Dict[str, str],
    last_valid_date: Dict[str, pd.Timestamp],
    market: str,
    tw_odd_lot: bool,
    round_trip_cost: Optional[float],
    weekly_us_tax: float,
    delist_return: float,
) -> tuple[pd.Series, List[float]]:
    held: Set[str] = set()
    wk_ret: Dict[pd.Timestamp, float] = {}
    turnovers: List[float] = []
    final_date = dates[-1]
    for i, d in enumerate(dates):
        grp = by_date[d]
        cost = 0.0
        if i % rebalance_every == offset or not held:
            new_set = _select_cohort_target(grp, score_col, top_k)
            if held and "is_limit_down" in grp.columns:
                locked = set(grp.loc[grp["is_limit_down"].fillna(False).astype(bool), "ticker"])
                new_set = new_set | (held & locked)
            c_val, turn = _compute_rebalance_cost(new_set, held, mkt_map, market, tw_odd_lot, round_trip_cost)
            if held:
                cost = c_val
                turnovers.append(turn)
            held = new_set
        ret_lookup = grp.set_index("ticker")["fwd_ret_1w_open"].to_dict()
        stk_rets, delisted_now = [], set()
        for sym in held:
            r_val = ret_lookup.get(sym, np.nan)
            if pd.notna(r_val) and np.isfinite(float(r_val)):
                r_net = float(r_val) - (weekly_us_tax if mkt_map.get(sym, market.upper()) == "US" else 0.0)
                stk_rets.append(r_net)
            elif d < final_date and (last_valid_date.get(sym) is None or last_valid_date[sym] < d):
                stk_rets.append(delist_return)
                delisted_now.add(sym)
        held -= delisted_now
        wk_ret[d] = (float(np.mean(stk_rets)) if stk_rets else 0.0) - cost
    return pd.Series(wk_ret), turnovers


def simulate_topk_twr(
    oos_df: pd.DataFrame,
    score_col: str,
    top_k: int = 40,
    rebalance_every: int = 4,
    round_trip_cost: Optional[float] = None,
    *,
    market: str = "ALL",
    tw_odd_lot: bool = False,
    us_div_withhold_annual: float = DEFAULT_US_DIV_WITHHOLD_ANNUAL,
    delist_return: float = DEFAULT_DELIST_RETURN,
) -> Dict[str, float]:
    """4-cohort staggered Top-K long-only weekly portfolio using real market costs, taxes & delisting."""
    empty = {k: np.nan for k in ("TWR_CAGR", "TWR_Sharpe", "TWR_MDD", "Avg_Weekly_Turnover", "Avg_Rebalance_Turnover", "Annual_Round_Trips")}
    if "fwd_ret_1w_open" not in oos_df.columns or oos_df.empty:
        return empty
    sub_df = oos_df if market.upper() == "ALL" else oos_df.loc[oos_df["market"] == market.upper()]
    dates = sorted(sub_df["date"].dropna().unique())
    if len(dates) < 8:
        return empty

    by_date = {d: grp for d, grp in sub_df.groupby("date")}
    mkt_map = dict(zip(sub_df["ticker"], sub_df["market"])) if "market" in sub_df.columns else {}
    last_valid_date = sub_df.dropna(subset=["fwd_ret_1w_open"]).groupby("ticker")["date"].max().to_dict()
    weekly_us_tax = us_div_withhold_annual / 52.0

    cohort_series, turnover_list = [], []
    for offset in range(rebalance_every):
        s_ret, turns = _run_single_cohort(
            offset, dates, by_date, score_col, top_k, rebalance_every, mkt_map,
            last_valid_date, market, tw_odd_lot, round_trip_cost, weekly_us_tax, delist_return,
        )
        cohort_series.append(s_ret)
        turnover_list.extend(turns)

    port_w = pd.concat(cohort_series, axis=1).mean(axis=1).dropna()
    if len(port_w) < 4:
        return empty
    nav = (1.0 + port_w).cumprod()
    sd = float(port_w.std())
    rebal_turn = float(np.mean(turnover_list)) if turnover_list else 0.0
    return {
        "TWR_CAGR": float(nav.iloc[-1] ** (52.0 / len(port_w)) - 1.0),
        "TWR_Sharpe": float(port_w.mean() / sd * np.sqrt(52.0)) if sd > 1e-12 else np.nan,
        "TWR_MDD": float((nav / nav.cummax() - 1.0).min()),
        "Avg_Weekly_Turnover": rebal_turn / rebalance_every,
        "Avg_Rebalance_Turnover": rebal_turn,
        "Annual_Round_Trips": rebal_turn * (52.0 / rebalance_every),
    }


def append_trial_log(
    log_path: Path,
    *,
    stage: str,
    market: str,
    model: str,
    change: str,
    ic: float,
    sharpe: float,
    mdd: float,
    t_nw: float = np.nan,
    cagr: float = np.nan,
    turnover: float = np.nan,
) -> int:
    """Append one trial record to `results/trial_log.csv` and return the new cumulative trial_id."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        df = pd.read_csv(log_path)
        next_id = int(df["trial_id"].max()) + 1 if not df.empty and "trial_id" in df.columns else 1
    else:
        df = pd.DataFrame()
        next_id = 1

    row = pd.DataFrame([{
        "trial_id": next_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "market": market,
        "model": model,
        "change": change,
        "oos_ic": round(float(ic), 5) if np.isfinite(ic) else np.nan,
        "t_nw": round(float(t_nw), 4) if np.isfinite(t_nw) else np.nan,
        "net_cagr": round(float(cagr), 5) if np.isfinite(cagr) else np.nan,
        "net_sharpe": round(float(sharpe), 4) if np.isfinite(sharpe) else np.nan,
        "mdd": round(float(mdd), 5) if np.isfinite(mdd) else np.nan,
        "rebal_turnover": round(float(turnover), 4) if np.isfinite(turnover) else np.nan,
    }])
    out = pd.concat([df, row], ignore_index=True)
    out.to_csv(log_path, index=False)
    return next_id
