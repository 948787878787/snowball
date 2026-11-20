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
from typing import Dict, List, Optional, Set, Tuple

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


def neutralize_score_panel(
    df: pd.DataFrame,
    score_col: str,
    *,
    risk_cols: Tuple[str, ...] = ("f_vol_60d", "f_low_risk"),
    industry_col: str = "industry",
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """Cross-sectionally regress `score_col` on risk factors + industry dummies per date, keeping residual."""
    target_out = out_col or f"{score_col}_neut"
    out = df.copy()
    out[target_out] = np.nan
    present_risk = [c for c in risk_cols if c in out.columns]
    grp_keys = ["date", "market"] if "market" in out.columns else ["date"]

    for _, grp in out.groupby(grp_keys):
        valid = grp.dropna(subset=[score_col])
        if len(valid) < 6:
            out.loc[valid.index, target_out] = valid[score_col]
            continue
        y = valid[score_col].to_numpy(dtype=float)
        parts = [np.ones((len(valid), 1), dtype=float)]
        if industry_col in valid.columns and valid[industry_col].nunique() > 1:
            dummies = pd.get_dummies(valid[industry_col].astype(str), drop_first=True, dtype=float).to_numpy()
            parts.append(dummies)
        for rc in present_risk:
            col_v = valid[rc].to_numpy(dtype=float)
            med = float(np.nanmedian(col_v)) if np.isfinite(col_v).any() else 0.0
            col_v = np.where(np.isfinite(col_v), col_v, med)
            if np.nanstd(col_v) > 1e-12:
                parts.append(col_v.reshape(-1, 1))
        X = np.hstack(parts)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[valid.index, target_out] = y - (X @ beta)
    return out


def _select_cohort_target(
    grp: pd.DataFrame,
    score_col: str,
    top_k: int,
    *,
    held: Optional[Set[str]] = None,
    max_industry_weight: Optional[float] = None,
    buffer_mult: float = 1.0,
) -> Set[str]:
    ranked = grp.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    if "is_limit_up" in ranked.columns:
        is_lu = ranked["is_limit_up"].fillna(False).astype(bool)
        keep_held = ranked["ticker"].isin(held) if held else False
        ranked = ranked[~is_lu | keep_held]
    k = min(top_k, max(4, len(ranked) // 10))
    if k <= 0 or ranked.empty:
        return set()

    has_ind = max_industry_weight is not None and max_industry_weight > 0 and "industry" in ranked.columns
    max_per_ind = max(1, int(np.floor(k * float(max_industry_weight) + 1e-9))) if has_ind else k
    ind_counts: Dict[str, int] = {}
    selected: Set[str] = set()
    tickers = ranked["ticker"].tolist()
    inds = ranked["industry"].astype(str).tolist() if has_ind else [""] * len(tickers)

    # Buffer pass: retain currently held names still inside top int(k * buffer_mult)
    if held and buffer_mult > 1.0:
        hold_cutoff = min(len(tickers), max(k, int(round(k * float(buffer_mult)))))
        for sym, ind in zip(tickers[:hold_cutoff], inds[:hold_cutoff]):
            if sym in held and ind_counts.get(ind, 0) < max_per_ind:
                selected.add(sym)
                ind_counts[ind] = ind_counts.get(ind, 0) + 1
                if len(selected) >= k:
                    return selected

    for sym, ind in zip(tickers, inds):
        if sym not in selected and ind_counts.get(ind, 0) < max_per_ind:
            selected.add(sym)
            ind_counts[ind] = ind_counts.get(ind, 0) + 1
            if len(selected) >= k:
                break
    return selected


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
    offset: int, dates: List[pd.Timestamp], by_date: Dict[pd.Timestamp, pd.DataFrame], score_col: str,
    top_k: int, rebalance_every: int, mkt_map: Dict[str, str], last_valid_date: Dict[str, pd.Timestamp],
    market: str, tw_odd_lot: bool, round_trip_cost: Optional[float], weekly_us_tax: float, delist_return: float,
    max_industry_weight: Optional[float] = None, buffer_mult: float = 1.0,
) -> tuple[pd.Series, List[float]]:
    held, wk_ret, turnovers, final_date = set(), {}, [], dates[-1]
    for i, d in enumerate(dates):
        grp, cost = by_date[d], 0.0
        if i % rebalance_every == offset or not held:
            new_set = _select_cohort_target(grp, score_col, top_k, held=held, max_industry_weight=max_industry_weight, buffer_mult=buffer_mult)
            if held and "is_limit_down" in grp.columns:
                new_set = new_set | (held & set(grp.loc[grp["is_limit_down"].fillna(False).astype(bool), "ticker"]))
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
                stk_rets.append(float(r_val) - (weekly_us_tax if mkt_map.get(sym, market.upper()) == "US" else 0.0))
            elif d < final_date and (last_valid_date.get(sym) is None or last_valid_date[sym] < d):
                stk_rets.append(delist_return)
                delisted_now.add(sym)
        held -= delisted_now
        wk_ret[d] = (float(np.mean(stk_rets)) if stk_rets else 0.0) - cost
    return pd.Series(wk_ret), turnovers


def _apply_ex_ante_vol_target(
    port_w: pd.Series, vol_target_annual: Optional[float], lookback_weeks: int = 12,
) -> tuple[pd.Series, float]:
    """Scale weekly returns using strictly lagged (t-1) realized volatility (zero look-ahead)."""
    if vol_target_annual is None or vol_target_annual <= 0:
        return port_w, 1.0
    trailing_vol = port_w.rolling(lookback_weeks, min_periods=4).std().shift(1) * np.sqrt(52.0)
    weights = (float(vol_target_annual) / trailing_vol.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0).fillna(1.0)
    return port_w * weights, float(weights.mean())


def simulate_topk_twr(
    oos_df: pd.DataFrame, score_col: str, top_k: int = 40, rebalance_every: int = 4,
    round_trip_cost: Optional[float] = None, *, market: str = "ALL", tw_odd_lot: bool = False,
    us_div_withhold_annual: float = DEFAULT_US_DIV_WITHHOLD_ANNUAL, delist_return: float = DEFAULT_DELIST_RETURN,
    max_industry_weight: Optional[float] = None, buffer_mult: float = 1.0,
    neutralize_scores: bool = False, vol_target_annual: Optional[float] = None,
) -> Dict[str, float]:
    """4-cohort staggered Top-K long-only weekly portfolio with Stage 3 risk & turnover controls."""
    keys = ("TWR_CAGR", "TWR_Sharpe", "TWR_MDD", "Avg_Weekly_Turnover", "Avg_Rebalance_Turnover", "Annual_Round_Trips", "Realized_Annual_Vol", "Avg_Exposure")
    empty = {k: np.nan for k in keys}
    if "fwd_ret_1w_open" not in oos_df.columns or oos_df.empty:
        return empty
    sub_df = oos_df if market.upper() == "ALL" else oos_df.loc[oos_df["market"] == market.upper()]
    settled = sub_df.dropna(subset=["fwd_ret_1w_open"])
    if settled.empty:
        return empty
    last_settled_date = settled["date"].max()
    sub_df = sub_df.loc[sub_df["date"] <= last_settled_date]
    dates = sorted(sub_df["date"].dropna().unique())
    if len(dates) < 8:
        return empty
    eff_col = f"{score_col}_neut" if neutralize_scores else score_col
    if neutralize_scores:
        sub_df = neutralize_score_panel(sub_df, score_col, out_col=eff_col)
    by_date = {d: grp for d, grp in sub_df.groupby("date")}
    mkt_map = dict(zip(sub_df["ticker"], sub_df["market"])) if "market" in sub_df.columns else {}
    last_valid_date = sub_df.dropna(subset=["fwd_ret_1w_open"]).groupby("ticker")["date"].max().to_dict()
    weekly_us_tax = us_div_withhold_annual / 52.0

    cohort_series, turnover_list = [], []
    for offset in range(rebalance_every):
        s_ret, turns = _run_single_cohort(
            offset, dates, by_date, eff_col, top_k, rebalance_every, mkt_map, last_valid_date,
            market, tw_odd_lot, round_trip_cost, weekly_us_tax, delist_return, max_industry_weight, buffer_mult,
        )
        cohort_series.append(s_ret)
        turnover_list.extend(turns)

    raw_w = pd.concat(cohort_series, axis=1).mean(axis=1).dropna()
    if len(raw_w) < 4:
        return empty
    port_w, avg_exp = _apply_ex_ante_vol_target(raw_w, vol_target_annual)
    nav, sd, rebal_turn = (1.0 + port_w).cumprod(), float(port_w.std()), (float(np.mean(turnover_list)) if turnover_list else 0.0)
    return {
        "TWR_CAGR": float(nav.iloc[-1] ** (52.0 / len(port_w)) - 1.0),
        "TWR_Sharpe": float(port_w.mean() / sd * np.sqrt(52.0)) if sd > 1e-12 else np.nan,
        "TWR_MDD": float((nav / nav.cummax() - 1.0).min()),
        "Avg_Weekly_Turnover": rebal_turn / rebalance_every,
        "Avg_Rebalance_Turnover": rebal_turn,
        "Annual_Round_Trips": rebal_turn * (52.0 / rebalance_every),
        "Realized_Annual_Vol": float(sd * np.sqrt(52.0)),
        "Avg_Exposure": avg_exp,
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
