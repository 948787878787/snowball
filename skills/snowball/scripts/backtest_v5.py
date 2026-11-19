"""v5 Institutional Audit & Personal Cash-Flow Allocator Backtest.

Directly addresses every major criticism from referee & code-review audits:
1. Honest labeling & Survivorship Bias disclosure:
   - 2020-2023 = In-Sample (Design); 2024-2026 = Out-of-Sample (Semi-contaminated Holdout).
   - Supports optional point-in-time `membership_mask` in `staggered_4w_eval()` for survivorship-free tests.
2. Apples-to-apples total-return benchmarks (Equal-Weight Universe TR + High-Yield TR),
   eliminating price-only ^TWII dividend mismatch.
3. Mathematical fixes to v5 factors:
   - Residual Momentum (Blitz et al. 2011: 26w skipping 4w cumulative Beta-adjusted alpha).
   - Full-group 12w industry alpha (fixing -1/(G-1)*r_i leave-one-out reversal artifact).
   - Cyclical-Normalized (5-year rolling median for cyclical sectors) + Industry-Neutral Value.
   - Drop PeerRev_4w (weight = 0).
4. Realistic Execution & Anti-Look-Ahead Controls:
   - Next-trading-day Open execution (`exec_open_w`, Monday open fill after Friday close signal),
     eliminating same-bar Friday-close execution bias.
   - Unadjusted nominal price (`raw_close_w`) for `<= max_price` affordability filter, preventing
     future-dividend look-ahead from backward-adjusted prices.
   - 4 staggered overlapping Jegadeesh-Titman cohorts, Newey-West lags = 6, Jensen's Alpha,
     and Sector Concentration (avg sectors & HHI).
5. Retail Cash-Flow Simulator (10,000 TWD/month DCA, Buy-Only + Zero Realized Loss + Trim Winners
   > 2x Target, with 0.30% odd-lot slippage, 0.1425% min 1 TWD commission, 0.30% sell tax):
   - Computes true unitized Time-Weighted Return (TWR) NAV drawdown (immune to deposit dilution).
   - Computes Core+Sat 60/40 combined drawdown strictly on the joint weekly portfolio NAV curve
     (never linearly averaging two separate MDDs).

Usage: .venv/bin/python backtest_v5.py
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import data
import evaluate as ev
import features as ft
from backtest_v3 import DESIGN, HOLDOUT_START, candidates as v3_candidates, load_panels
from backtest_v4 import build_v4_panels
from universe import PB_GROUPS, PEER_GROUPS, group_of

OUT_DIR = Path(__file__).parent / "results"
CYCLICAL_GROUPS = frozenset({"container_shipping", "memory", "steel", "petrochemical", "airline"})
YIELD_CAP_PCT = 8.0
EY_CAP = 0.15
NW_LAGS = 6
ROUND_TRIP_COST_INST = 0.00585
ODD_LOT_COMM = 0.001425
ODD_LOT_MIN_FEE = 1.0
ODD_LOT_SLIPPAGE = 0.0030
SELL_TAX = 0.0030


def _group_mean_incl(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Full peer-group equal-weight mean (including self) to avoid -1/(G-1) small-group reversal artifact."""
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for members in groups.values():
        cols = [c for c in members if c in df.columns]
        if not cols:
            continue
        m = df[cols].mean(axis=1)
        for c in cols:
            out[c] = m
    return out


def _group_zscore(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Within-peer-group standardized score (industry-neutral)."""
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for members in groups.values():
        cols = [c for c in members if c in df.columns]
        if len(cols) < 2:
            continue
        block = df[cols]
        mu = block.mean(axis=1)
        sd = block.std(axis=1).replace(0.0, np.nan)
        z = block.sub(mu, axis=0).div(sd, axis=0).clip(-3.0, 3.0)
        for c in cols:
            out[c] = z[c]
    return out


def _next_trading_day_open_weekly(open_d: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """For each weekly Friday signal timestamp `dt`, return the first trading day's Open strictly after `dt`."""
    open_sorted = open_d.sort_index()
    idx = open_sorted.index
    rows = []
    for dt in weeks:
        pos = int(idx.searchsorted(dt, side="right"))
        if pos < len(open_sorted):
            rows.append(open_sorted.iloc[pos])
        else:
            rows.append(pd.Series(np.nan, index=open_sorted.columns))
    return pd.DataFrame(rows, index=weeks, columns=open_sorted.columns)


def compute_twr_nav(weekly_val: pd.Series, weekly_cf: pd.Series) -> pd.Series:
    """Compute true unitized Time-Weighted Return (TWR) NAV series from weekly end values and external deposits.

    Deposits occur at the start of the period (before that period's return), so:
    `r_t = end_val_t / (prev_end_val + deposit_t) - 1`.
    Completely eliminates fake drawdowns caused by dividing portfolio value by cumulative deposits.
    """
    nav_vals = []
    nav = 1.0
    prev_val = 0.0
    for val, cf in zip(weekly_val.to_numpy(dtype=float), weekly_cf.to_numpy(dtype=float)):
        denom = prev_val + cf
        if denom > 0.0:
            nav *= val / denom
        nav_vals.append(nav)
        prev_val = val
    return pd.Series(nav_vals, index=weekly_val.index, dtype=float)


def build_v5_panels(p: dict) -> dict:
    """Construct clean point-in-time v5 factors, nominal prices, next-day opens, and weekly total returns."""
    pr_d = data.load_prices()
    close_d = pr_d["close"]
    raw_close_d = pr_d.get("raw_close", close_d)
    open_d = pr_d.get("open", close_d)
    weeks = p["weeks"]
    groups = p["groups"]
    cols = p["alpha"].columns
    gmap = group_of()

    close_w = close_d.resample("W-FRI").last().reindex(weeks)[cols]
    raw_close_w = raw_close_d.resample("W-FRI").last().reindex(weeks)[cols]
    exec_open_w = _next_trading_day_open_weekly(open_d[cols], weeks)
    ret_1w = close_w.pct_change(fill_method=None)
    mkt_1w = pr_d["index"].resample("W-FRI").last().reindex(weeks).pct_change(fill_method=None)
    alpha = p["alpha"]

    val_df = data.load_valuation()

    def _piv_w(col: str) -> pd.DataFrame:
        w = val_df.pivot_table(index="date", columns="stock_id", values=col, aggfunc="last")
        return w.resample("W-FRI").last().reindex(weeks).reindex(columns=cols)

    yld = _piv_w("dividend_yield")
    per = _piv_w("PER")
    pbr = _piv_w("PBR")
    ey = (1.0 / per.where(per > 0)).fillna(0.0).where(per.notna())
    bp = (1.0 / pbr.where(pbr > 0)).where(pbr.notna())

    ey_capped = ey.clip(upper=EY_CAP)
    yld_capped = yld.clip(upper=YIELD_CAP_PCT)
    ey_norm = ey_capped.copy()
    yld_norm = yld_capped.copy()
    for c in cols:
        if gmap.get(c) in CYCLICAL_GROUPS:
            ey_norm[c] = ey_capped[c].rolling(260, min_periods=52).median()
            yld_norm[c] = yld_capped[c].rolling(260, min_periods=52).median()

    core_val_raw = pd.DataFrame(
        {c: (bp[c] if gmap.get(c) in PB_GROUPS else ey_norm[c]) for c in cols},
        index=weeks,
    )
    val_ind_neutral = _group_zscore(core_val_raw, groups).fillna(0.0)
    val_yield_norm = ft.cs_zscore(yld_norm).fillna(0.0)
    f_norm_val = ft.cs_zscore(val_ind_neutral + 0.5 * val_yield_norm).where(yld.notna() | core_val_raw.notna())

    res_mom_raw = alpha.shift(4).rolling(22, min_periods=12).sum()
    f_res_mom = ft.cs_zscore(res_mom_raw)

    alpha_12w = alpha.rolling(12, min_periods=6).sum()
    f_ind_mom_full = ft.cs_zscore(_group_mean_incl(alpha_12w, groups))

    f_fin = p["finstmt"]
    mask = p["zrel"].notna()

    v5_core = ft.cs_zscore(
        f_res_mom.fillna(0.0) + f_fin.fillna(0.0) + f_norm_val.fillna(0.0)
    ).where(mask)

    v5_income = ft.cs_zscore(
        0.5 * f_res_mom.fillna(0.0)
        + 0.5 * f_ind_mom_full.fillna(0.0)
        + f_fin.fillna(0.0)
        + 1.5 * f_norm_val.fillna(0.0)
    ).where(mask)

    ew_ret_1w = ret_1w.mean(axis=1)
    hy_mask = yld.rank(axis=1, pct=True) >= 0.80
    hy_ret_1w = ret_1w.where(hy_mask.shift(1).fillna(False)).mean(axis=1)

    return {
        "close_w": close_w,
        "raw_close_w": raw_close_w,
        "exec_open_w": exec_open_w,
        "ret_1w": ret_1w,
        "mkt_1w": mkt_1w,
        "ew_ret_1w": ew_ret_1w,
        "hy_ret_1w": hy_ret_1w,
        "yld": yld,
        "yld_norm": yld_norm,
        "per": per,
        "f_res_mom": f_res_mom,
        "f_ind_mom_full": f_ind_mom_full,
        "f_norm_val": f_norm_val,
        "f_fin": f_fin,
        "v5_core": v5_core,
        "v5_income": v5_income,
    }


def staggered_4w_eval(
    signal: pd.DataFrame,
    ret_1w: pd.DataFrame,
    ew_ret_1w: pd.Series,
    mkt_1w: pd.Series,
    target_4w: pd.DataFrame,
    start: str,
    end,
    buy_frac: float = 0.20,
    hold_frac: float = 0.20,
    close_w: Optional[pd.DataFrame] = None,
    exec_open_w: Optional[pd.DataFrame] = None,
    membership_mask: Optional[pd.DataFrame] = None,
) -> dict:
    """Jegadeesh-Titman 4-cohort staggered portfolio evaluation with true weekly mark-to-market MDD
    and Next-Trading-Day Open execution when `close_w` and `exec_open_w` are provided.
    """
    sig_all = signal.loc[start:end]
    if membership_mask is not None:
        aligned_mask = membership_mask.reindex(index=sig_all.index, columns=sig_all.columns).fillna(False).astype(bool)
        sig_all = sig_all.where(aligned_mask)

    ret_all = ret_1w.loc[start:end]
    ew_all = ew_ret_1w.loc[start:end]
    tgt_all = target_4w.loc[start:end]
    close_all = close_w.loc[start:end] if close_w is not None else None
    open_all = exec_open_w.loc[start:end] if exec_open_w is not None else None
    gmap = group_of()

    ic_vals, top_spread, bot_spread = [], [], []
    dates = sig_all.index
    for dt in dates:
        if dt not in tgt_all.index:
            continue
        a, b = sig_all.loc[dt], tgt_all.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() < 15 or a[m].nunique() <= 1:
            continue
        ic_vals.append(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))
        k = int(np.floor(m.sum() * 0.20))
        order = a[m].sort_values().index
        mu = b[m].mean()
        top_spread.append(b[order[-k:]].mean() - mu)
        bot_spread.append(mu - b[order[:k]].mean())

    ic_s = pd.Series(ic_vals)
    ic_nw_t = ev.newey_west_t(ic_s, lags=NW_LAGS)
    long_side_ann = float(pd.Series(top_spread).mean() * 13.0)
    short_side_ann = float(pd.Series(bot_spread).mean() * 13.0)

    cohort_net_rets = []
    cohort_turns = []
    n_sectors_list, hhi_list = [], []
    use_next_open = close_all is not None and open_all is not None

    for offset in range(4):
        held = set()
        wk_rets = {}
        for i in range(len(dates) - 1):
            dt = dates[i]
            nxt = dates[i + 1]
            cost = 0.0
            new_entries = set()
            exited = set()
            if i % 4 == offset or not held:
                a = sig_all.loc[dt].dropna().sort_values(ascending=False)
                n = len(a)
                k_buy = int(np.floor(n * buy_frac))
                k_hold = int(np.floor(n * hold_frac))
                if k_buy >= 1:
                    top_hold = set(a.index[:k_hold])
                    new_held = [s for s in held if s in top_hold]
                    for s in a.index:
                        if len(new_held) >= k_buy:
                            break
                        if s not in new_held:
                            new_held.append(s)
                    new_set = set(new_held)
                    if held:
                        new_entries = new_set - held
                        exited = held - new_set
                        turn = len(new_entries) / max(len(new_set), 1)
                        cohort_turns.append(turn)
                        cost = turn * ROUND_TRIP_COST_INST
                    else:
                        new_entries = set(new_set)
                    held = new_set

                    sec_counts = pd.Series([gmap.get(s, "other") for s in held]).value_counts()
                    w_sec = sec_counts / sec_counts.sum()
                    n_sectors_list.append(len(sec_counts))
                    hhi_list.append(float((w_sec ** 2).sum()))

            if held:
                if use_next_open and new_entries and dt in open_all.index:
                    per_stock_rets = []
                    retained = held - new_entries
                    if retained:
                        per_stock_rets.extend(ret_all.loc[nxt, list(retained)].dropna().tolist())
                    ent_open = open_all.loc[dt, list(new_entries)]
                    ent_close = close_all.loc[nxt, list(new_entries)]
                    r_ent = (ent_close / ent_open.replace(0.0, np.nan) - 1.0).dropna()
                    if exited:
                        ex_open = open_all.loc[dt, list(exited)]
                        ex_prev = close_all.loc[dt, list(exited)]
                        gap_exit = (ex_open / ex_prev.replace(0.0, np.nan) - 1.0).dropna()
                        gap_mean = float(gap_exit.mean()) if len(gap_exit) else 0.0
                        r_ent = (1.0 + gap_mean) * (1.0 + r_ent) - 1.0
                    per_stock_rets.extend(r_ent.tolist())
                    gross_r = float(np.mean(per_stock_rets)) if per_stock_rets else 0.0
                else:
                    r_next = ret_all.loc[nxt, list(held)].dropna()
                    gross_r = float(r_next.mean()) if len(r_next) else 0.0
                wk_rets[nxt] = gross_r - cost
        cohort_net_rets.append(pd.Series(wk_rets))

    port_w = pd.concat(cohort_net_rets, axis=1).mean(axis=1).dropna()
    ew_w = ew_all.reindex(port_w.index).dropna()
    port_w = port_w.reindex(ew_w.index)
    ex_w = port_w - ew_w

    eq = (1.0 + port_w).cumprod()
    ew_eq = (1.0 + ew_w).cumprod()
    n_weeks = len(port_w)
    cagr = float(eq.iloc[-1] ** (52.0 / n_weeks) - 1.0)
    ew_cagr = float(ew_eq.iloc[-1] ** (52.0 / n_weeks) - 1.0)
    true_mdd = float((eq / eq.cummax() - 1.0).min())
    ew_mdd = float((ew_eq / ew_eq.cummax() - 1.0).min())
    sharpe = float(port_w.mean() / port_w.std() * np.sqrt(52.0)) if port_w.std() > 0 else np.nan

    cov_pm = np.cov(port_w, ew_w)[0, 1]
    var_m = np.var(ew_w, ddof=1)
    beta_ew = float(cov_pm / var_m) if var_m > 0 else 1.0
    jensen_w = port_w - beta_ew * ew_w
    jensen_ann = float(jensen_w.mean() * 52.0)
    jensen_t = float(ev.newey_west_t(jensen_w, lags=NW_LAGS))

    ex_ann = float(ex_w.mean() * 52.0)
    ex_t = float(ev.newey_west_t(ex_w, lags=NW_LAGS))

    return {
        "IC": float(ic_s.mean()),
        "IC_t_NW6": float(ic_nw_t),
        "Long_Gross": long_side_ann,
        "Short_Gross": short_side_ann,
        "Net_Ex_EW": ex_ann,
        "Net_Ex_t": ex_t,
        "Beta_EW": beta_ew,
        "Jensen_Ann": jensen_ann,
        "Jensen_t": jensen_t,
        "Port_CAGR": cagr,
        "EW_CAGR": ew_cagr,
        "True_MDD": true_mdd,
        "EW_MDD": ew_mdd,
        "Sharpe": sharpe,
        "Turn_4w": float(np.mean(cohort_turns)) if cohort_turns else 0.0,
        "Avg_Sectors": float(np.mean(n_sectors_list)) if n_sectors_list else np.nan,
        "Sector_HHI": float(np.mean(hhi_list)) if hhi_list else np.nan,
    }


def simulate_personal_cashflow(
    signal: pd.DataFrame,
    close_w: pd.DataFrame,
    yld: pd.DataFrame,
    start: str,
    end,
    monthly_inflow: float = 10000.0,
    target_k: int = 10,
    max_price: float = 800.0,
    trim_winners: bool = True,
    mode: str = "model",
    exec_open_w: Optional[pd.DataFrame] = None,
    raw_close_w: Optional[pd.DataFrame] = None,
    return_series: bool = False,
) -> dict:
    """Simulate realistic Taiwan retail investor: fixed monthly DCA, Buy-Only, Zero Realized Loss.

    Anti-look-ahead & mathematical fixes:
    - Affordability filter (`<= max_price`) uses `raw_close_w` (unadjusted point-in-time nominal price)
      when provided, preventing future-dividend look-ahead.
    - Buys & winner-trims execute at `exec_open_w` (next trading day's Open after Friday signal `dt`)
      when provided, eliminating same-bar Friday-close execution bias.
    - Drawdown (`True_TWR_MDD_%`) is computed on the unitized Time-Weighted Return (TWR) NAV curve,
      eliminating deposit-dilution artifacts from `Value / Contributed`.
    """
    sig_sub = signal.loc[start:end]
    px_sub = close_w.ffill().loc[start:end]
    raw_px_sub = (raw_close_w.ffill().loc[start:end] if raw_close_w is not None else px_sub)
    open_sub = (exec_open_w.reindex(px_sub.index).combine_first(px_sub) if exec_open_w is not None else px_sub)
    yld_sub = yld.ffill().loc[start:end]
    dates = sig_sub.index

    shares = {}
    cost_basis = {}
    total_contributed = 0.0
    cash = 0.0
    n_buys = 0
    n_winner_trims = 0
    weekly_vals = []
    weekly_cfs = []
    max_single_w = 0.0
    w_target = 1.0 / target_k

    for i, dt in enumerate(dates):
        prices_close = px_sub.loc[dt].dropna()
        # Trade execution uses next-trading-day Open when executing a Friday signal; mark-to-market uses Friday Close.
        # On week `i` when `i % 4 == 0`, deposits and orders decided at `dt` execute at `open_sub.loc[dt]` (next Monday open),
        # and are then marked to market at the following week's close (or end of bar `i` if same-bar fallback).
        # Specifically: evaluate signal at prior Friday `dates[i-1]` -> execute at `open_sub.loc[dates[i-1]]` (Monday of week `i`)
        # -> mark-to-market at Friday `dates[i]`! That way signal, fill, and week-end MTM are 100% chronologically ordered!
        cf_this_week = 0.0
        if i % 4 == 0:
            sig_dt = dates[i - 1] if (exec_open_w is not None and i > 0) else dt
            fill_prices = open_sub.loc[sig_dt].dropna() if exec_open_w is not None else prices_close
            nom_prices = raw_px_sub.loc[sig_dt].dropna()

            mv_map_fill = {s: shares[s] * float(fill_prices[s]) for s in shares if s in fill_prices and shares[s] > 0}
            port_mv = sum(mv_map_fill.values())

            cash += monthly_inflow
            total_contributed += monthly_inflow
            cf_this_week = monthly_inflow
            total_eq = port_mv + cash

            # Step 1: Trim overweight winners ONLY (never sell at a loss!)
            if trim_winners and total_eq > 0:
                for s, mv in list(mv_map_fill.items()):
                    avg_c = cost_basis[s] / shares[s] if shares[s] > 0 else 1e9
                    curr_p = float(fill_prices[s])
                    if mv / total_eq > 2.0 * w_target and curr_p > avg_c:
                        excess_val = mv - 1.5 * w_target * total_eq
                        sell_sh = excess_val / curr_p
                        exec_p = curr_p * (1.0 - ODD_LOT_SLIPPAGE)
                        gross_proc = sell_sh * exec_p
                        fee = max(ODD_LOT_MIN_FEE, gross_proc * ODD_LOT_COMM)
                        tax = gross_proc * SELL_TAX
                        net_proc = gross_proc - fee - tax
                        frac_sold = sell_sh / shares[s]
                        cost_basis[s] *= (1.0 - frac_sold)
                        shares[s] -= sell_sh
                        cash += net_proc
                        n_winner_trims += 1
                mv_map_fill = {s: shares[s] * float(fill_prices[s]) for s in shares if s in fill_prices and shares[s] > 0}
                port_mv = sum(mv_map_fill.values())
                total_eq = port_mv + cash

            # Step 2: Select buy candidate (affordability checked strictly on unadjusted nominal price `nom_prices`)
            affordable_idx = nom_prices[nom_prices <= max_price].index.intersection(fill_prices.index)
            if mode == "ew_bench":
                investable = cash * (1.0 - ODD_LOT_COMM)
                per_stock = investable / len(fill_prices)
                for s, p_val in fill_prices.items():
                    sh = per_stock / float(p_val)
                    shares[s] = shares.get(s, 0.0) + sh
                    cost_basis[s] = cost_basis.get(s, 0.0) + per_stock
                cash = 0.0
                n_buys += 1
            elif mode == "hy_bench":
                y_row = yld_sub.loc[sig_dt].reindex(fill_prices.index).dropna().sort_values(ascending=False)
                top_hy = y_row.index[: max(1, int(len(y_row) * 0.20))]
                investable = cash * (1.0 - ODD_LOT_COMM)
                per_stock = investable / len(top_hy)
                for s in top_hy:
                    p_val = float(fill_prices[s])
                    sh = per_stock / p_val
                    shares[s] = shares.get(s, 0.0) + sh
                    cost_basis[s] = cost_basis.get(s, 0.0) + per_stock
                cash = 0.0
                n_buys += 1
            else:
                a = sig_sub.loc[sig_dt].reindex(affordable_idx).dropna().sort_values(ascending=False)
                k_top = max(5, int(np.floor(len(a) * 0.20)))
                cands_top = a.index[:k_top]
                best_s, best_prio = None, -1e9
                for s in cands_top:
                    w_now = (mv_map_fill.get(s, 0.0) / total_eq) if total_eq > 0 else 0.0
                    prio = float(a[s]) + 1.0 * (w_target - w_now) / w_target
                    if prio > best_prio:
                        best_prio = prio
                        best_s = s
                if best_s is not None and cash > 500:
                    exec_p = float(fill_prices[best_s]) * (1.0 + ODD_LOT_SLIPPAGE)
                    fee = max(ODD_LOT_MIN_FEE, cash * ODD_LOT_COMM)
                    net_cash = cash - fee
                    sh = net_cash / exec_p
                    shares[best_s] = shares.get(best_s, 0.0) + sh
                    cost_basis[best_s] = cost_basis.get(best_s, 0.0) + cash
                    cash = 0.0
                    n_buys += 1

        # End-of-week Friday Close mark-to-market
        mv_map = {s: shares[s] * float(prices_close[s]) for s in shares if s in prices_close and shares[s] > 0}
        end_val = sum(mv_map.values()) + cash
        if end_val > 0 and mv_map and total_contributed >= 10.0 * monthly_inflow:
            max_single_w = max(max_single_w, max(mv_map.values()) / end_val)
        weekly_vals.append(end_val)
        weekly_cfs.append(cf_this_week)

    val_s = pd.Series(weekly_vals, index=dates, dtype=float)
    cf_s = pd.Series(weekly_cfs, index=dates, dtype=float)
    nav_s = compute_twr_nav(val_s, cf_s)
    true_twr_mdd = float((nav_s / nav_s.cummax() - 1.0).min())
    terminal_val = float(val_s.iloc[-1])
    total_roi = (terminal_val / total_contributed - 1.0) * 100.0 if total_contributed > 0 else 0.0

    last_y = yld_sub.iloc[-1]
    last_p = px_sub.iloc[-1]
    end_mvs = {s: shares[s] * float(last_p[s]) for s in shares if s in last_p and pd.notna(last_p[s]) and shares[s] > 0}
    end_tot = sum(end_mvs.values())
    port_yield = sum(
        (mv / end_tot) * float(last_y.get(s, 0.0))
        for s, mv in end_mvs.items()
        if pd.notna(last_y.get(s, np.nan))
    ) if end_tot > 0 else 0.0

    res = {
        "Contributed_TWD": round(total_contributed, 0),
        "Terminal_TWD": round(terminal_val, 0),
        "Profit_TWD": round(terminal_val - total_contributed, 0),
        "Total_ROI_%": round(total_roi, 1),
        "True_TWR_MDD_%": round(true_twr_mdd * 100.0, 1),
        "Max_Stock_Wt_%": round(max_single_w * 100.0, 1),
        "End_Yield_%": round(port_yield, 2),
        "Holdings_Count": sum(1 for v in end_mvs.values() if v > 100),
        "Winner_Trims": n_winner_trims,
    }
    if return_series:
        res["_weekly_val"] = val_s
        res["_weekly_cf"] = cf_s
        res["_twr_nav"] = nav_s
    return res


def combine_cashflow_simulations(core_res: dict, sat_res: dict, label: str = "Core+Sat 60/40") -> dict:
    """Combine two retail cash-flow simulations via their exact joint weekly portfolio NAV curve.

    Never linearly averages two separate MDDs.
    """
    comb_val = core_res["_weekly_val"].add(sat_res["_weekly_val"], fill_value=0.0)
    comb_cf = core_res["_weekly_cf"].add(sat_res["_weekly_cf"], fill_value=0.0)
    comb_nav = compute_twr_nav(comb_val, comb_cf)
    comb_mdd = float((comb_nav / comb_nav.cummax() - 1.0).min())

    comb_contrib = core_res["Contributed_TWD"] + sat_res["Contributed_TWD"]
    comb_term = float(comb_val.iloc[-1])
    sat_share_end = sat_res["Terminal_TWD"] / comb_term if comb_term > 0 else 0.0
    return {
        "Strategy": label,
        "Contributed_TWD": round(comb_contrib, 0),
        "Terminal_TWD": round(comb_term, 0),
        "Profit_TWD": round(comb_term - comb_contrib, 0),
        "Total_ROI_%": round((comb_term / comb_contrib - 1.0) * 100.0, 1) if comb_contrib > 0 else 0.0,
        "True_TWR_MDD_%": round(comb_mdd * 100.0, 1),
        "Max_Stock_Wt_%": round(sat_share_end * sat_res["Max_Stock_Wt_%"], 1),
        "End_Yield_%": round(
            (core_res["Terminal_TWD"] * core_res["End_Yield_%"] + sat_res["Terminal_TWD"] * sat_res["End_Yield_%"]) / comb_term,
            2,
        ) if comb_term > 0 else 0.0,
        "Holdings_Count": sat_res["Holdings_Count"],
        "Winner_Trims": sat_res["Winner_Trims"],
    }


def run() -> None:
    print(
        "[NOTICE] Survivorship Bias Disclosure: The built-in 84-stock Taiwan universe is curated ex-post "
        "and omits delisted firms. Expect live net alpha in a point-in-time Top-150 universe to be lower.\n"
        "[AUDIT] Execution: Next-Trading-Day Open | Affordability Filter: Unadjusted Nominal Close | "
        "Retail MDD: Unitized TWR NAV."
    )
    p = load_panels()
    v3 = v3_candidates(p)
    v4 = build_v4_panels(p)
    v5 = build_v5_panels(p)

    models = {
        "v3_smooth4w": v3["composite_v3_smooth4w"],
        "v4_income": v4["v4_income"],
        "v5_core (ResMom+Fin+NormVal)": v5["v5_core"],
        "v5_income (+0.5 IndMomFull)": v5["v5_income"],
    }

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 22)

    for label, key, start, end in [
        ("IN-SAMPLE DESIGN (2020-01 .. 2023-12) | 4-COHORT STAGGERED, NEXT-DAY OPEN, TRUE WEEKLY MDD", "v5_insample_2020_2023.csv", DESIGN[0], DESIGN[1]),
        ("OUT-OF-SAMPLE HOLDOUT (2024-01 .. 2026-11) | 4-COHORT STAGGERED, NEXT-DAY OPEN, TRUE WEEKLY MDD", "v5_holdout_2024_2026.csv", HOLDOUT_START, None),
    ]:
        rows = []
        for name, sig in models.items():
            st = staggered_4w_eval(
                sig,
                v5["ret_1w"],
                v5["ew_ret_1w"],
                v5["mkt_1w"],
                p["target_4w"],
                start,
                end,
                0.20,
                0.20,
                close_w=v5["close_w"],
                exec_open_w=v5["exec_open_w"],
            )
            rows.append({"model": name, **st})
        df = pd.DataFrame(rows)
        OUT_DIR.mkdir(exist_ok=True)
        df.to_csv(OUT_DIR / key, index=False)
        print(f"\n=== {label} ===")
        print(df.round(4).to_string(index=False))

    for label, key, start, end in [
        ("RETAIL 10,000 TWD/MO ZERO-LOSS SIMULATOR | HOLDOUT (2024-01 .. 2026-11, ~2.9 yrs)", "v5_retail_sim_holdout.csv", HOLDOUT_START, None),
        ("RETAIL 10,000 TWD/MO ZERO-LOSS SIMULATOR | 7-YEAR FULL PATH (2020-01 .. 2026-11, note IS+OOS)", "v5_retail_sim_7yr.csv", DESIGN[0], None),
    ]:
        sim_kw = dict(exec_open_w=v5["exec_open_w"], raw_close_w=v5["raw_close_w"])
        core_res = simulate_personal_cashflow(
            v5["v5_core"], v5["close_w"], v5["yld"], start, end,
            monthly_inflow=6000.0, mode="hy_bench", return_series=True, **sim_kw,
        )
        sat_res = simulate_personal_cashflow(
            v5["v5_core"], v5["close_w"], v5["yld"], start, end,
            monthly_inflow=4000.0, mode="model", return_series=True, **sim_kw,
        )
        comb_row = combine_cashflow_simulations(
            core_res, sat_res, label="Core+Sat 60/40 (6,000 HighYield + 4,000 v5_core)"
        )
        sim_rows = [
            {"Strategy": "Bench 1: Equal-Weight 84-Stock TR DCA", **simulate_personal_cashflow(v5["v5_income"], v5["close_w"], v5["yld"], start, end, mode="ew_bench", **sim_kw)},
            {"Strategy": "Bench 2: High-Yield Top-20% TR DCA (0056 proxy)", **simulate_personal_cashflow(v5["v5_income"], v5["close_w"], v5["yld"], start, end, mode="hy_bench", **sim_kw)},
            {"Strategy": "v4_income (Buy-Only + Trim Winners >2x)", **simulate_personal_cashflow(v4["v4_income"], v5["close_w"], v5["yld"], start, end, mode="model", **sim_kw)},
            {"Strategy": "v5_core (Buy-Only + Trim Winners >2x)", **simulate_personal_cashflow(v5["v5_core"], v5["close_w"], v5["yld"], start, end, mode="model", **sim_kw)},
            {"Strategy": "v5_income (Buy-Only + Trim Winners >2x)", **simulate_personal_cashflow(v5["v5_income"], v5["close_w"], v5["yld"], start, end, mode="model", **sim_kw)},
            comb_row,
        ]
        sim_df = pd.DataFrame(sim_rows)
        sim_df.to_csv(OUT_DIR / key, index=False)
        print(f"\n=== {label} ===")
        print(sim_df.to_string(index=False))

    dt = v5["v5_income"].index[-1]
    print(f"\n=== LATEST TOP 12 STOCKS ({dt.strftime('%Y-%m-%d')}) | v5_income ===")
    top = v5["v5_income"].loc[dt].dropna().sort_values(ascending=False).head(12)
    t_rows = []
    for sid, sc in top.items():
        t_rows.append({
            "sid": sid,
            "close": v5["close_w"].loc[dt, sid],
            "raw_close": v5["raw_close_w"].loc[dt, sid],
            "yld_%": v5["yld"].loc[dt, sid],
            "yld_5y_norm": v5["yld_norm"].loc[dt, sid],
            "v5_inc": sc,
            "res_mom": v5["f_res_mom"].loc[dt, sid],
            "ind_mom": v5["f_ind_mom_full"].loc[dt, sid],
            "fin": v5["f_fin"].loc[dt, sid],
            "norm_val": v5["f_norm_val"].loc[dt, sid],
        })
    print(pd.DataFrame(t_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    run()
