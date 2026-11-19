"""v4 Deep Optimization: Value, Momentum, Fundamentals, and Peer-Timing Everywhere.

Strictly designed on the Design Period (2020-01 .. 2023-12) without peeking at Holdout,
then frozen and evaluated once on the Holdout Period (2024-01 .. latest).

Key upgrades over v3:
1. Add Industry Momentum (12w peer mean alpha, Moskowitz & Grinblatt 1999 JF).
2. Add Medium-Term Stock Momentum (26w skipping last 4w, Jegadeesh & Titman 1993 JF).
3. Upgrade Valuation from peer-only relative value to Market-Wide Absolute Value
   (Dividend Yield + Earnings Yield + 0.5 * Book-to-Price, Asness et al. 2013 JF / Gormsen & Lazarus 2023 JF).
4. Replace high-turnover 1w peer reversal with 4w peer reversal (Hameed & Mian 2015 JFQA).
5. Drop noisy, high-turnover institutional flow (F_Flow -> 0).
6. Add Sell-Buffer Portfolio Rebalancing (Buy Top 20%, Hold until Top 40%, Novy-Marx & Velikov 2016 RFS)
   to slash round-trip trading costs in Taiwan equities (0.585%).

Usage: .venv/bin/python backtest_v4.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import data
import evaluate as ev
import features as ft
from backtest_v3 import DESIGN, HOLDOUT_START, candidates as v3_candidates, load_panels, long_only

OUT_DIR = Path(__file__).parent / "results"


def build_v4_panels(p: dict) -> dict:
    """Build point-in-time v4 pillars using only data available at or before week t."""
    pr_d = data.load_prices()
    close_d = pr_d["close"]
    weeks = p["weeks"]
    groups = p["groups"]
    cols = p["alpha"].columns

    close_w = close_d.resample("W-FRI").last().reindex(weeks)[cols]
    mkt_w = pr_d["index"].resample("W-FRI").last().reindex(weeks)
    ret_4w_raw = close_w.shift(-4) / close_w - 1.0
    mkt_4w_raw = mkt_w.shift(-4) / mkt_w - 1.0

    alpha = p["alpha"]

    # 1. Market-wide Absolute Value: Dividend Yield + Earnings Yield + 0.5 * Book-to-Price
    val_df = data.load_valuation()

    def _piv_w(col: str) -> pd.DataFrame:
        w = val_df.pivot_table(index="date", columns="stock_id", values=col, aggfunc="last")
        return w.resample("W-FRI").last().reindex(weeks).reindex(columns=cols)

    yld, per, pbr = _piv_w("dividend_yield"), _piv_w("PER"), _piv_w("PBR")
    ey = (1.0 / per.where(per > 0)).fillna(0.0).where(per.notna())
    bp = (1.0 / pbr.where(pbr > 0)).where(pbr.notna())
    abs_val = ft.cs_zscore(
        ft.cs_zscore(yld).fillna(0.0)
        + ft.cs_zscore(ey).fillna(0.0)
        + 0.5 * ft.cs_zscore(bp).fillna(0.0)
    ).where(yld.notna())

    # 2. Industry Momentum (12-week peer group alpha, excluding self)
    alpha_12w = alpha.rolling(12, min_periods=6).sum()
    ind_mom_12w = ft.cs_zscore(ft.peer_mean_excl(alpha_12w, groups))

    # 3. Medium-Term Stock Momentum (26-week return skipping most recent 4 weeks)
    mom_26_4 = ft.cs_zscore(close_w.shift(4) / close_w.shift(26) - 1.0)

    # 4. 4-Week Peer Reversion (buy relative laggard within peer group)
    alpha_4w = alpha.rolling(4, min_periods=2).sum()
    rel_4w = alpha_4w - ft.peer_mean_excl(alpha_4w, groups)
    zrel_4w = ft.ts_zscore(rel_4w, 52, 26)
    peer_rev_4w = ft.cs_zscore(-zrel_4w)

    # 5. Fundamentals (point-in-time trailing 3m revenue YoY + acceleration)
    fin = p["finstmt"]

    mask = p["zrel"].notna()

    # Pre-committed v4 formulas (frozen from Design Period 2020-2023)
    v4_balanced = ft.cs_zscore(
        ind_mom_12w.fillna(0.0)
        + mom_26_4.fillna(0.0)
        + fin.fillna(0.0)
        + abs_val.fillna(0.0)
        + 0.5 * peer_rev_4w.fillna(0.0)
    ).where(mask)

    v4_income = ft.cs_zscore(
        ind_mom_12w.fillna(0.0)
        + 0.5 * mom_26_4.fillna(0.0)
        + fin.fillna(0.0)
        + 1.5 * abs_val.fillna(0.0)
        + 0.5 * peer_rev_4w.fillna(0.0)
    ).where(mask)

    return {
        "ind_mom_12w": ind_mom_12w,
        "mom_26_4": mom_26_4,
        "abs_val": abs_val,
        "peer_rev_4w": peer_rev_4w,
        "fin": fin,
        "v4_balanced": v4_balanced,
        "v4_income": v4_income,
        "ret_4w_raw": ret_4w_raw,
        "mkt_4w_raw": mkt_4w_raw,
        "yld": yld,
        "per": per,
        "close_w": close_w,
    }


def long_only_buffer(
    signal: pd.DataFrame,
    target_alpha: pd.DataFrame,
    raw_ret: pd.DataFrame,
    mkt_ret: pd.Series,
    horizon: int = 4,
    buy_frac: float = 0.20,
    hold_frac: float = 0.40,
) -> dict:
    """Long-only top-basket backtest with Sell-Buffer hysteresis (Buy Top 20%, Hold Top 40%)."""
    sig = bt._subsample(signal, horizon)
    tgt = bt._subsample(target_alpha, horizon)
    raw = bt._subsample(raw_ret, horizon)
    mkt = bt._subsample(mkt_ret, horizon)

    held = set()
    ex_gross, ex_net, port_net, mkt_list, turn_list = [], [], [], [], []
    dates = []

    for dt in sig.index:
        a = sig.loc[dt]
        b = tgt.loc[dt]
        r = raw.loc[dt] if dt in raw.index else pd.Series(dtype=float)
        m = a.notna() & b.notna() & r.notna()
        valid_a = a[m].sort_values(ascending=False)
        n = len(valid_a)
        k_buy = int(np.floor(n * buy_frac))
        k_hold = int(np.floor(n * hold_frac))
        if k_buy < 1:
            continue

        top_hold = set(valid_a.index[:k_hold])
        new_held = [s for s in held if s in top_hold]
        for s in valid_a.index:
            if len(new_held) >= k_buy:
                break
            if s not in new_held:
                new_held.append(s)

        new_set = set(new_held)
        turn = 0.0 if not held else len(new_set - held) / max(len(new_set), 1)
        held = new_set

        g_alpha = b[list(held)].mean() - b[m].mean()
        n_alpha = g_alpha - turn * bt.ROUND_TRIP_COST
        p_net = r[list(held)].mean() - turn * bt.ROUND_TRIP_COST
        m_r = float(mkt.loc[dt]) if dt in mkt.index and pd.notna(mkt.loc[dt]) else np.nan

        ex_gross.append(g_alpha)
        ex_net.append(n_alpha)
        port_net.append(p_net)
        mkt_list.append(m_r)
        turn_list.append(turn)
        dates.append(dt)

    s_g = pd.Series(ex_gross, index=dates)
    s_n = pd.Series(ex_net, index=dates)
    s_p = pd.Series(port_net, index=dates)
    s_m = pd.Series(mkt_list, index=dates)
    per = bt.WEEKS_PER_YEAR / horizon

    # Equity curve & max drawdown on realized raw portfolio returns
    eq = (1.0 + s_p).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else np.nan
    cagr = float(eq.iloc[-1] ** (per / len(eq)) - 1.0) if len(eq) else np.nan
    mkt_eq = (1.0 + s_m.dropna()).cumprod()
    mkt_cagr = float(mkt_eq.iloc[-1] ** (per / len(mkt_eq)) - 1.0) if len(mkt_eq) else np.nan
    sharpe = float(s_p.mean() / s_p.std() * np.sqrt(per)) if s_p.std() > 0 else np.nan

    return {
        "buf_gross_ann": float(s_g.mean() * per),
        "buf_net_ann": float(s_n.mean() * per),
        "buf_net_t": float(s_n.mean() / s_n.std() * np.sqrt(len(s_n))) if s_n.std() > 0 else np.nan,
        "buf_turn": float(np.mean(turn_list)),
        "port_cagr": cagr,
        "twii_cagr": mkt_cagr,
        "port_sharpe": sharpe,
        "port_mdd": mdd,
    }


def _pairs(sig: pd.DataFrame, tgt: pd.DataFrame):
    j = pd.concat([sig.stack(), tgt.stack()], axis=1, keys=["s", "t"]).dropna()
    return j["s"].to_numpy(), (j["t"] > 0).astype(float).to_numpy()


def calibrated_accuracy_4w(signal: pd.DataFrame, target: pd.DataFrame) -> dict:
    design_idx = target.loc[DESIGN[0]:DESIGN[1]].index
    x, y = _pairs(bt._subsample(signal.loc[design_idx], 4), bt._subsample(target.loc[design_idx], 4))
    a, k = ev.fit_logistic(x, y)
    hold_idx = target.loc[HOLDOUT_START:].index
    prob = ft.sigmoid(a + k * signal.loc[hold_idx])
    return {"a": a, "kappa": k, **bt.prob_stats(prob, target.loc[hold_idx], 4)}


def evaluate_models(models: dict, p: dict, v4: dict, start: str, end) -> pd.DataFrame:
    tgt = p["target_4w"].loc[start:end]
    raw = v4["ret_4w_raw"].loc[start:end]
    mkt = v4["mkt_4w_raw"].loc[start:end]
    rows = []
    for name, sig in models.items():
        s = sig.loc[start:end]
        st = bt.signal_stats(s, tgt, 4)
        lo = long_only(s, tgt, 4)
        bf = long_only_buffer(s, tgt, raw, mkt, 4, buy_frac=0.20, hold_frac=0.40)
        rows.append({
            "model": name,
            "IC_mean": st["IC_mean"],
            "IC_t_NW": st["IC_t_NW"],
            "IC_hit": st["IC_hit"],
            "LO_net_ann": lo["LO_excess_net_ann"],
            "LO_net_t": lo["LO_net_t"],
            "LO_turn": lo["LO_turnover"],
            **bf,
        })
    return pd.DataFrame(rows)


def run() -> None:
    p = load_panels()
    v3 = v3_candidates(p)
    v4 = build_v4_panels(p)

    models = {
        "v2_peer": v3["peer_v2"],
        "v3_composite": v3["composite_v3"],
        "v3_smooth4w": v3["composite_v3_smooth4w"],
        "v4_balanced": v4["v4_balanced"],
        "v4_income": v4["v4_income"],
    }

    design = evaluate_models(models, p, v4, *DESIGN)
    holdout = evaluate_models(models, p, v4, HOLDOUT_START, None)
    full = evaluate_models(models, p, v4, DESIGN[0], None)

    acc_rows = []
    for name in ("v3_smooth4w", "v4_balanced", "v4_income"):
        acc_rows.append({"model": name, **calibrated_accuracy_4w(models[name], p["target_4w"])})
    acc = pd.DataFrame(acc_rows)

    OUT_DIR.mkdir(exist_ok=True)
    design.to_csv(OUT_DIR / "v4_design_2020_2023.csv", index=False)
    holdout.to_csv(OUT_DIR / "v4_holdout_2024_2026.csv", index=False)
    full.to_csv(OUT_DIR / "v4_full_2020_2026.csv", index=False)
    acc.to_csv(OUT_DIR / "v4_holdout_accuracy.csv", index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)
    cols = [
        "model", "IC_mean", "IC_t_NW", "IC_hit",
        "LO_net_ann", "LO_net_t", "LO_turn",
        "buf_net_ann", "buf_net_t", "buf_turn",
        "port_cagr", "twii_cagr", "port_sharpe", "port_mdd",
    ]
    print("=== DESIGN PERIOD (2020-2023) | 4-WEEK HORIZON ===")
    print(design[cols].round(4).to_string(index=False))
    print("\n=== FROZEN HOLDOUT PERIOD (2024-01 .. 2026-11) | 4-WEEK HORIZON ===")
    print(holdout[cols].round(4).to_string(index=False))
    print("\n=== FULL OOS PERIOD (2020-01 .. 2026-11) | 4-WEEK HORIZON ===")
    print(full[cols].round(4).to_string(index=False))
    print("\n=== HOLDOUT PROBABILITY ACCURACY (4-WEEK) ===")
    print(acc.round(4).to_string(index=False))

    # Print latest top 15 rankings for v4_income and v4_balanced
    dt = v4["v4_income"].index[-1]
    print(f"\n=== LATEST TOP 15 STOCKS ({dt.strftime('%Y-%m-%d')}) | v4_income (高息價值優化版) ===")
    top = v4["v4_income"].loc[dt].dropna().sort_values(ascending=False).head(15)
    t_rows = []
    for sid, sc in top.items():
        t_rows.append({
            "sid": sid,
            "close": v4["close_w"].loc[dt, sid],
            "yield_%": v4["yld"].loc[dt, sid],
            "PER": v4["per"].loc[dt, sid],
            "v4_income": sc,
            "v4_bal": v4["v4_balanced"].loc[dt, sid],
            "ind_mom": v4["ind_mom_12w"].loc[dt, sid],
            "mom_26_4": v4["mom_26_4"].loc[dt, sid],
            "fin": v4["fin"].loc[dt, sid],
            "abs_val": v4["abs_val"].loc[dt, sid],
            "peer_rev4w": v4["peer_rev_4w"].loc[dt, sid],
        })
    print(pd.DataFrame(t_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    run()
