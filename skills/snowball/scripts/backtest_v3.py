"""v3 optimization with a design/holdout split.

Design period (2020-01 .. 2023-12): look at term-level results, choose the v3 formula.
Holdout period (2024-01 .. latest): evaluate the chosen formula once, with frozen parameters.
Usage: .venv/bin/python backtest_v3.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import data
import evaluate as ev
import features as ft
from panels import build_panels

OUT_DIR = Path(__file__).parent / "results"
DESIGN = ("2020-01-01", "2023-12-31")
HOLDOUT_START = "2024-01-01"
SMOOTH_WEEKS = 4


def load_panels(force_refresh: bool = False) -> dict:
    """Build or load cached weekly factor panels with TTL, schema-version, and universe-hash invalidation."""
    def _build() -> dict:
        return build_panels(
            data.load_prices(force_refresh=force_refresh),
            data.load_revenue(force_refresh=force_refresh),
            data.load_valuation(force_refresh=force_refresh),
            data.load_institutional(force_refresh=force_refresh),
        )

    return data.cached_load(
        "panels",
        _build,
        ttl_seconds=data.DEFAULT_DAILY_TTL_SECONDS,
        schema_version=f"{data.CACHE_SCHEMA_VERSION}-panels",
        force_refresh=force_refresh,
    )


def candidates(p: dict) -> dict:
    zrel, s, v = p["zrel"], p["s"].fillna(0.5), p["v"]
    mask = zrel.notna()
    rev_pure = -zrel
    peer_v3 = ft.cs_zscore(rev_pure + v.fillna(0.0)).where(mask)
    peer_v3_s = ft.cs_zscore(-s * zrel + v.fillna(0.0)).where(mask)
    fin = p["finstmt"]
    flow = p["flow"]
    comp_v3 = ft.cs_zscore(fin.fillna(0) + peer_v3.fillna(0) + 0.5 * flow.fillna(0)).where(mask)
    comp_v3_smooth = comp_v3.rolling(SMOOTH_WEEKS, min_periods=1).mean().where(mask)
    return {
        "peer_v2": p["peer_v2"],
        "peer_v3_pure_reversion": ft.cs_zscore(rev_pure),
        "peer_v3_s_gated": peer_v3_s,
        "peer_v3": peer_v3,
        "finstmt_revenue": fin,
        "composite_v3": comp_v3,
        "composite_v3_smooth4w": comp_v3_smooth,
    }


def long_only(signal, target, horizon, frac=0.2) -> dict:
    sig, tgt = bt._subsample(signal, horizon), bt._subsample(target, horizon)
    top = {}
    for dt in sig.index:
        a, b = sig.loc[dt], tgt.loc[dt]
        m = a.notna() & b.notna()
        k = int(np.floor(m.sum() * frac))
        if k >= 1:
            top[dt] = b[a[m].sort_values().index[-k:]].mean() - b[m].mean()
    top = pd.Series(top)
    turn = bt._basket_turnover(sig, frac, True).reindex(top.index).fillna(0)
    net = top - turn * bt.ROUND_TRIP_COST
    per = bt.WEEKS_PER_YEAR / horizon
    return {"LO_excess_gross_ann": top.mean() * per, "LO_excess_net_ann": net.mean() * per,
            "LO_net_t": net.mean() / net.std() * np.sqrt(len(net)), "LO_turnover": turn.mean()}


def evaluate_period(cands, p, start, end) -> pd.DataFrame:
    rows = []
    for h, key in ((1, "target_1w"), (4, "target_4w")):
        tgt = p[key].loc[start:end]
        for name, sig in cands.items():
            s = sig.loc[start:end]
            rows.append({"horizon_w": h, "signal": name, **bt.signal_stats(s, tgt, h), **long_only(s, tgt, h)})
    return pd.DataFrame(rows)


def calibrated_accuracy(signal, p, horizon, key) -> dict:
    tgt = p[key]
    design_dates = tgt.loc[DESIGN[0]:DESIGN[1]].index
    x, y = _pairs(bt._subsample(signal.loc[design_dates], horizon), bt._subsample(tgt.loc[design_dates], horizon))
    a, k = ev.fit_logistic(x, y)
    hold = tgt.loc[HOLDOUT_START:].index
    prob = ft.sigmoid(a + k * signal.loc[hold])
    return {"a": a, "kappa": k, **bt.prob_stats(prob, tgt.loc[hold], horizon)}


def _pairs(sig, tgt):
    j = pd.concat([sig.stack(), tgt.stack()], axis=1, keys=["s", "t"]).dropna()
    return j["s"].to_numpy(), (j["t"] > 0).astype(float).to_numpy()


def run():
    p = load_panels()
    cands = candidates(p)
    design = evaluate_period(cands, p, *DESIGN)
    holdout = evaluate_period(cands, p, HOLDOUT_START, None)
    acc = pd.DataFrame([{"horizon_w": h, "model": "composite_v3_smooth4w",
                         **calibrated_accuracy(cands["composite_v3_smooth4w"], p, h, k)}
                        for h, k in ((1, "target_1w"), (4, "target_4w"))])
    design.to_csv(OUT_DIR / "v3_design_2020_2023.csv", index=False)
    holdout.to_csv(OUT_DIR / "v3_holdout_2024_2026.csv", index=False)
    acc.to_csv(OUT_DIR / "v3_holdout_accuracy.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["horizon_w", "signal", "IC_mean", "IC_t_NW", "IC_hit", "LS_net_ann", "LO_excess_gross_ann",
            "LO_excess_net_ann", "LO_net_t", "LO_turnover"]
    print("=== DESIGN 2020-2023 ===")
    print(design[cols].round(4).to_string(index=False))
    print("=== HOLDOUT 2024-01 .. latest ===")
    print(holdout[cols].round(4).to_string(index=False))
    print(acc.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
