"""Walk-forward backtest of the v2 formula on Taiwan peer groups.

Usage: .venv/bin/python backtest.py
Writes results/report.md and results/*.csv. All numbers are out-of-sample (OOS) unless labeled.
"""

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import composite as cp
import data
import evaluate as ev
from panels import build_panels

LOG = logging.getLogger("backtest")
OUT_DIR = Path(__file__).parent / "results"
TRAIN_WEEKS = 104
ROUND_TRIP_COST = 0.001425 * 2 + 0.003  # fee both sides + securities transaction tax
WEEKS_PER_YEAR = 52


def _subsample(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return df.iloc[::horizon] if horizon > 1 else df


def _basket_turnover(signal: pd.DataFrame, frac: float, top: bool) -> pd.Series:
    prev, out = None, {}
    for dt in signal.index:
        a = signal.loc[dt].dropna()
        k = int(np.floor(len(a) * frac))
        if k < 1:
            continue
        names = set(a.sort_values().index[-k:] if top else a.sort_values().index[:k])
        out[dt] = 1.0 if prev is None else 1 - len(names & prev) / k
        prev = names
    return pd.Series(out, dtype=float)


def signal_stats(signal: pd.DataFrame, target: pd.DataFrame, horizon: int) -> dict:
    sig, tgt = _subsample(signal, horizon), _subsample(target, horizon)
    ic = ev.rank_ic(sig, tgt, min_names=20)
    summ = ev.ic_summary(ic)
    ls = ev.long_short(sig, tgt, 0.2)
    turn = _basket_turnover(sig, 0.2, True).add(_basket_turnover(sig, 0.2, False), fill_value=0)
    ls_net = ls - turn.reindex(ls.index).fillna(0) * ROUND_TRIP_COST
    periods = WEEKS_PER_YEAR / horizon
    return {
        "IC_mean": summ["mean"], "IC_t_NW": summ["t_nw"], "IC_hit": summ["hit"], "weeks": summ["n"],
        "LS_gross_ann": ls.mean() * periods, "LS_net_ann": ls_net.mean() * periods,
        "LS_net_t": ls_net.mean() / ls_net.std() * np.sqrt(len(ls_net)) if len(ls_net) > 2 else np.nan,
    }


def prob_stats(prob: pd.DataFrame, target: pd.DataFrame, horizon: int, conf: float = 0.05) -> dict:
    pr, tg = _subsample(prob, horizon), _subsample(target, horizon)
    j = pd.concat([pr.stack(), tg.stack()], axis=1, keys=["p", "t"]).dropna()
    p, o = j["p"].to_numpy(), (j["t"] > 0).astype(float).to_numpy()
    rel, res, unc = ev.murphy(p, o)
    hi = np.abs(p - 0.5) >= conf
    return {
        "N": len(p), "base_rate": o.mean(), "hit_rate": float(((p > 0.5) == (o > 0.5)).mean()),
        "Brier": ev.brier(p, o), "BSS_vs_0.25": ev.brier_skill(p, o),
        "BSS_vs_base": ev.brier_skill(p, o, ref=o.mean() * (1 - o.mean())),
        "REL": rel, "RES": res, "UNC": unc,
        "conf_coverage": float(hi.mean()),
        "conf_hit_rate": float(((p[hi] > 0.5) == (o[hi] > 0.5)).mean()) if hi.any() else np.nan,
        "p_min": float(p.min()), "p_max": float(p.max()),
    }


def yearly_ic(signal, target, horizon) -> pd.Series:
    ic = ev.rank_ic(_subsample(signal, horizon), _subsample(target, horizon), min_names=20)
    return ic.groupby(ic.index.year).mean()


def peer_terms(p: dict) -> dict:
    s = p["s"].fillna(0.5)
    return {
        "peer_t1_leader_gap": (p["follower"] * p["zg"]).where(p["zrel"].notna()),
        "peer_t2_reversion": -s * p["zrel"],
        "peer_t3_competitive": (1 - s) * p["zrel"].clip(upper=0),
        "peer_t4_valuation": p["v"].where(p["zrel"].notna()),
    }


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(exist_ok=True)
    prices = data.load_prices()
    p = build_panels(prices, data.load_revenue(), data.load_valuation(), data.load_institutional())
    pillars = cp.pillar_panels(p)
    valid = p["peer_v2"].notna().sum(axis=1)
    first = valid[valid >= 40].index.min()
    oos_start = p["weeks"][p["weeks"].get_loc(first) + TRAIN_WEEKS]
    LOG.info("stocks=%d weeks=%d first_valid=%s oos_start=%s", p["alpha"].shape[1], len(p["weeks"]), first.date(), oos_start.date())

    singles = {
        "peer_v1_draft": p["peer_v1"], "peer_v2": p["peer_v2"], **peer_terms(p),
        "finstmt_revenue": p["finstmt"], "flow_institutional": p["flow"],
        "macro_lagged": p["macro"], "chain_us_semis": p["chain"],
        "reversal_only": -p["zabn"],
    }
    prior_s, prior_p = cp.prior_model(p, pillars)
    results, prob_rows, yearly, weight_logs = [], [], {}, {}
    for h, tgt_key in ((1, "target_1w"), (4, "target_4w")):
        tgt = p[tgt_key]
        oos = tgt.index >= oos_start
        wf_s, wf_p, wlog = cp.walk_forward(p, pillars, tgt, oos_start, horizon=h)
        weight_logs[h] = wlog
        models = {**singles, "composite_prior_doc_v2": prior_s, "composite_walkforward": wf_s}
        for name, sig in models.items():
            results.append({"horizon_w": h, "signal": name, **signal_stats(sig[oos], tgt[oos], h)})
        prob_rows.append({"horizon_w": h, "model": "prior_doc_v2", **prob_stats(prior_p[oos], tgt[oos], h)})
        prob_rows.append({"horizon_w": h, "model": "walkforward_calibrated", **prob_stats(wf_p[oos], tgt[oos], h)})
        for name in ("peer_v1_draft", "peer_v2", "composite_prior_doc_v2", "composite_walkforward"):
            yearly[f"{name}_{h}w"] = yearly_ic(models[name][oos], tgt[oos], h)

    res = pd.DataFrame(results)
    probs = pd.DataFrame(prob_rows)
    yearly_df = pd.DataFrame(yearly)
    res.to_csv(OUT_DIR / "signal_stats.csv", index=False)
    probs.to_csv(OUT_DIR / "prob_stats.csv", index=False)
    yearly_df.to_csv(OUT_DIR / "yearly_ic.csv")
    for h, wlog in weight_logs.items():
        wlog.to_csv(OUT_DIR / f"weights_{h}w.csv", index=False)
    meta = {"run_date": date.today().isoformat(), "stocks": int(p["alpha"].shape[1]),
            "first_valid": str(first.date()), "oos_start": str(oos_start.date()),
            "last_week": str(p["weeks"][-1].date()), "groups": {g: list(m) for g, m in p["groups"].items()}}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 220)
    print(json.dumps(meta, ensure_ascii=False)[:400])
    print(res.round(4).to_string(index=False))
    print(probs.round(4).to_string(index=False))
    print(yearly_df.round(3).to_string())
    print(weight_logs[1].tail(3).to_string(index=False))
    print(weight_logs[4].tail(3).to_string(index=False))


if __name__ == "__main__":
    run()
