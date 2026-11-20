"""Stage 5 Econometric Validation: Multi-Seed Ensemble, CPCV & Regime Attribution (Snowball v7).

References:
- Gu, Kelly, & Xiu (2020), *Review of Financial Studies*: Regularized linear + tree ensembles
  with economic monotonicity constraints for cross-sectional asset pricing.
- Lopez de Prado (2018), *Advances in Financial Machine Learning*, Ch. 12: Combinatorial
  Purged Cross-Validation (CPCV) with symmetric purge & embargo around test boundaries.
- Bailey, Borwein, Lopez de Prado, & Zhu (2014), *J. Computational Finance*: Probability of
  Backtest Overfitting (PBO) & Deflated Sharpe Ratio.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import evaluate as ev
import portfolio_sim as psim
import train_walkforward as twf


def generate_cpcv_splits(
    dates: Sequence[pd.Timestamp],
    *,
    n_groups: int = 6,
    k_test_groups: int = 2,
    purge_weeks: int = 5,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    """Generate C(n_groups, k_test_groups) purged combinatorial train/test date splits."""
    uniq = sorted(pd.Timestamp(d) for d in pd.unique(pd.Series(dates).dropna()))
    blocks = [list(b) for b in np.array_split(uniq, n_groups) if len(b) > 0]
    purge_td = pd.Timedelta(weeks=purge_weeks)
    splits: List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]] = []

    for test_idx in combinations(range(len(blocks)), k_test_groups):
        test_set = set(test_idx)
        te_dates = sorted(d for i in test_idx for d in blocks[i])
        test_ranges = [(min(blocks[i]), max(blocks[i])) for i in test_idx]
        tr_dates = []
        for i, blk in enumerate(blocks):
            if i in test_set:
                continue
            for d in blk:
                too_close = any((lo - purge_td) <= d <= (hi + purge_td) for lo, hi in test_ranges)
                if not too_close:
                    tr_dates.append(d)
        splits.append((sorted(tr_dates), te_dates))
    return splits


def build_rank_ensemble(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    *,
    out_col: str = "pred_ensemble",
) -> pd.DataFrame:
    """Compute equal-weight cross-sectional percentile rank ensemble across `score_cols`."""
    out = df.copy()
    grp_keys = ["date", "market"] if "market" in out.columns else ["date"]
    ranks = [out.groupby(grp_keys)[c].rank(pct=True) for c in score_cols if c in out.columns]
    if not ranks:
        out[out_col] = np.nan
        return out
    out[out_col] = pd.concat(ranks, axis=1).mean(axis=1)
    return out


def compute_portfolio_weekly_returns(
    oos_df: pd.DataFrame,
    score_col: str,
    *,
    top_k: int = 20,
    rebalance_every: int = 4,
    market: str = "TW",
    tw_odd_lot: bool = False,
    max_industry_weight: Optional[float] = 0.25,
    buffer_mult: float = 10.0,
    neutralize_scores: bool = True,
    vol_target_annual: Optional[float] = None,
) -> pd.Series:
    """Return the exact net weekly return series of the 4-cohort staggered Stage 3 portfolio."""
    sub_df = oos_df if market.upper() == "ALL" else oos_df.loc[oos_df["market"] == market.upper()]
    settled = sub_df.dropna(subset=["fwd_ret_1w_open"])
    if not settled.empty:
        sub_df = sub_df.loc[sub_df["date"] <= settled["date"].max()]
    dates = sorted(sub_df["date"].dropna().unique())
    eff_col = f"{score_col}_neut" if neutralize_scores else score_col
    if neutralize_scores:
        sub_df = psim.neutralize_score_panel(sub_df, score_col, out_col=eff_col)
    by_date = {d: grp for d, grp in sub_df.groupby("date")}
    mkt_map = dict(zip(sub_df["ticker"], sub_df["market"])) if "market" in sub_df.columns else {}
    last_valid = sub_df.dropna(subset=["fwd_ret_1w_open"]).groupby("ticker")["date"].max().to_dict()
    weekly_us_tax = psim.DEFAULT_US_DIV_WITHHOLD_ANNUAL / 52.0

    cohorts = []
    for off in range(rebalance_every):
        s_ret, _ = psim._run_single_cohort(
            off, dates, by_date, eff_col, top_k, rebalance_every, mkt_map, last_valid,
            market, tw_odd_lot, None, weekly_us_tax, psim.DEFAULT_DELIST_RETURN,
            max_industry_weight, buffer_mult,
        )
        cohorts.append(s_ret)
    raw_w = pd.concat(cohorts, axis=1).mean(axis=1).dropna()
    port_w, _ = psim._apply_ex_ante_vol_target(raw_w, vol_target_annual)
    return port_w


def compute_excess_metrics(
    port_w: pd.Series,
    bench_w: pd.Series,
    *,
    lags: int = 4,
) -> Dict[str, float]:
    """Compute net excess return over benchmark, Newey-West t-stat, Net IR, and Excess MDD."""
    aligned = pd.concat([port_w.rename("port"), bench_w.rename("bench")], axis=1).dropna()
    if len(aligned) < 4:
        return {k: np.nan for k in ("Net_Excess_CAGR", "Net_Excess_Ann", "Excess_t_NW", "Net_IR", "Excess_MDD")}
    p, b = aligned["port"], aligned["bench"]
    ex = p - b
    p_nav, b_nav = (1.0 + p).cumprod(), (1.0 + b).cumprod()
    n_w = len(aligned)
    p_cagr = float(p_nav.iloc[-1] ** (52.0 / n_w) - 1.0)
    b_cagr = float(b_nav.iloc[-1] ** (52.0 / n_w) - 1.0)
    p_mdd = float((p_nav / p_nav.cummax() - 1.0).min())
    b_mdd = float((b_nav / b_nav.cummax() - 1.0).min())
    sd_ex = float(ex.std())
    if sd_ex <= 1e-12:
        t_nw, ir = 0.0, 0.0
    else:
        t_nw = float(ev.newey_west_t(ex, lags=max(4, int(lags))))
        ir = float(ex.mean() / sd_ex * np.sqrt(52.0))
    return {
        "Net_Excess_CAGR": p_cagr - b_cagr,
        "Net_Excess_Ann": float(ex.mean() * 52.0),
        "Excess_t_NW": t_nw,
        "Net_IR": ir,
        "Excess_MDD": p_mdd - b_mdd,
    }


def passes_v7_relative_gate(
    excess_t_nw: float,
    dsr_conservative: float,
    *,
    min_t_nw: float = 2.0,
    min_dsr: float = 0.95,
) -> bool:
    """Strict v7 T5 pass gate: Net excess over TW EW t_NW >= 2.0 AND conservative DSR >= 0.95."""
    return bool(
        np.isfinite(excess_t_nw)
        and np.isfinite(dsr_conservative)
        and float(excess_t_nw) >= min_t_nw
        and float(dsr_conservative) >= min_dsr
    )


def summarize_annual_regimes(port_w: pd.Series, ew_w: pd.Series) -> pd.DataFrame:
    """Attribute portfolio return, excess return, Sharpe, and MDD across calendar-year regimes."""
    aligned = pd.concat([port_w.rename("port"), ew_w.rename("ew")], axis=1).dropna()
    rows: List[Dict[str, float]] = []
    for yr, grp in aligned.groupby(aligned.index.year):
        if len(grp) < 4:
            continue
        p_nav = (1.0 + grp["port"]).cumprod()
        e_nav = (1.0 + grp["ew"]).cumprod()
        p_ret = float(p_nav.iloc[-1] - 1.0)
        e_ret = float(e_nav.iloc[-1] - 1.0)
        sd = float(grp["port"].std())
        rows.append({
            "year": int(yr),
            "n_weeks": int(len(grp)),
            "port_ret": p_ret,
            "ew_ret": e_ret,
            "excess_ret": p_ret - e_ret,
            "port_sharpe": float(grp["port"].mean() / sd * np.sqrt(52.0)) if sd > 1e-12 else np.nan,
            "port_mdd": float((p_nav / p_nav.cummax() - 1.0).min()),
        })
    return pd.DataFrame(rows)


def run_cpcv_ridge_lgbm(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "target_rank_mkt_4w",
    *,
    n_groups: int = 6,
    k_test_groups: int = 2,
    purge_weeks: int = 5,
    ridge_alpha: float = 500.0,
    lgbm_seeds: Sequence[int] = (42, 137, 2026),
) -> pd.DataFrame:
    """Run C(6,2)=15 Combinatorial Purged CV paths for Constrained Ridge + 3-Seed Monotone LightGBM."""
    df = panel.dropna(subset=[target_col]).copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    splits = generate_cpcv_splits(df["date"].unique(), n_groups=n_groups, k_test_groups=k_test_groups, purge_weeks=purge_weeks)
    f_list = list(feature_cols)
    rows: List[Dict[str, float]] = []

    for pid, (tr_dates, te_dates) in enumerate(splits, start=1):
        tr = df.loc[df["date"].isin(tr_dates)]
        te = df.loc[df["date"].isin(te_dates)].copy()
        if len(tr) < 500 or te.empty:
            continue
        X_tr, y_tr = tr[f_list].to_numpy(dtype=float), tr[target_col].to_numpy(dtype=float)
        X_te = te[f_list].to_numpy(dtype=float)

        r_int, r_coef = twf._fit_ridge(X_tr, y_tr, l2_alpha=ridge_alpha, feature_names=f_list)
        te["pred_ridge"] = twf._predict_ridge(X_te, r_int, r_coef)

        lgb_preds = []
        X_te_df = pd.DataFrame(X_te, columns=f_list)
        for sd in lgbm_seeds:
            m_lgb = twf._fit_lightgbm(X_tr, y_tr, f_list, use_gpu=False, seed=int(sd))
            lgb_preds.append(m_lgb.predict(X_te_df))
        te["pred_lgbm3"] = np.mean(lgb_preds, axis=0)
        te = build_rank_ensemble(te, ["pred_ridge", "pred_lgbm3"], out_col="pred_ens")

        ic_r = twf._compute_daily_ic_series(te, "pred_ridge", target_col, market_filter="TW")
        ic_l = twf._compute_daily_ic_series(te, "pred_lgbm3", target_col, market_filter="TW")
        ic_e = twf._compute_daily_ic_series(te, "pred_ens", target_col, market_filter="TW")
        rows.append({
            "path_id": pid,
            "n_test_weeks": int(te["date"].nunique()),
            "ridge_ic": float(ic_r.mean()),
            "lgbm3_ic": float(ic_l.mean()),
            "ens_ic": float(ic_e.mean()),
            "ens_hit": float((ic_e > 0).mean()),
        })
    return pd.DataFrame(rows)
