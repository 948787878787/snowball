"""One-shot run of the pre-registered ETF risk-control research (configs/etf_risk_v1.json).

Refuses unless the config is FROZEN with a code commit, and refuses a non-empty output directory.
Every arm is appended to results/trial_log.csv (stage ETF_RISK_V1). Research only: no orders.

Usage:
  python run_etf_risk_v1.py --config ../../../configs/etf_risk_v1.json \
      --trial-log ../../../results/trial_log.csv --out-dir ../../../../snowball_artifacts_etf_risk_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import etf_risk_data as rd
import etf_risk_eval as ee
import etf_risk_sim as rs
import portfolio_sim as psim


def require_frozen(cfg: Dict[str, Any]) -> None:
    if cfg.get("status") != "FROZEN" or not cfg.get("code_commit"):
        raise PermissionError("etf_risk_v1 runs only on a FROZEN config with code_commit set")


def load_market(cfg: Dict[str, Any]) -> pd.DataFrame:
    import yfinance as yf

    def close(sym: str, field: str) -> pd.Series:
        raw = yf.download(sym, start="2011-01-01", end="2026-11-21", progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            raise RuntimeError(f"No data for {sym}")
        s = raw[field]
        return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).dropna()

    d = cfg["data"]
    tw = close("006208.TW", "Adj Close")
    fx = rd.clean_reversing_spikes(close("TWD=X", "Close"), threshold=float(d["fx_spike_threshold"]))
    us = rd.us_sleeve_twd(close("SPY", "Adj Close"), fx, annual_fee=float(d["us_annual_drag"]))
    return build_prices(tw, us)


def build_prices(tw: pd.Series, us_twd: pd.Series) -> pd.DataFrame:
    days = pd.DatetimeIndex(tw.index)
    prices = pd.DataFrame({"tw": tw.to_numpy(dtype=float), "us": rs.align_strictly_before(us_twd, days).to_numpy()},
                          index=days)
    return prices.dropna()


def make_rules(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t, band, a = cfg["targets"], float(cfg["rebalance_band_abs"]), cfg["arms"]
    return {
        "S0_static": rs.StaticRule(t, band),
        "A1_trend": rs.TrendRule(t, int(a["A1_trend"]["sma_days"]), band),
        "A2_voltarget": rs.VolTargetRule(t, float(a["A2_voltarget"]["vol_target_annual"]),
                                         int(a["A2_voltarget"]["vol_window_days"]), band),
        "B_tw50_hold": rs.HoldRule({"tw": 1.0, "us": 0.0, "cash": 0.0}),
    }


def _first_signal(prices: pd.DataFrame, min_days: int = 252) -> pd.Timestamp:
    return rs.month_end_days(prices.index, prices.index[min_days - 1])[0]


def evaluate_arms(prices: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    start = _first_signal(prices)
    halves, costs = cfg["sample"]["halves"], cfg["costs"]
    sims = {k: rs.simulate(prices, r, start, float(cfg["data"]["cash_yield_annual"]), costs)
            for k, r in make_rules(cfg).items()}
    base = sims["S0_static"].nav
    out: Dict[str, Any] = {"first_signal": str(start.date()), "last_date": str(prices.index[-1].date()), "arms": {}}
    for k, sim in sims.items():
        years = (sim.nav.index[-1] - sim.nav.index[0]).days / 365.25
        out["arms"][k] = {**ee.describe(sim.nav, halves), "costs_paid": sim.costs_paid,
                          "annual_turnover": sim.traded_value / float(sim.nav.mean()) / years,
                          "n_trade_days": sim.n_trade_days,
                          "vs_S0_monthly": ee.monthly_excess(sim.nav, base) if k != "S0_static" else None}
        out["arms"][k]["sensitivity_cash"] = {
            str(y): ee.performance(rs.simulate(prices, make_rules(cfg)[k], start, y, costs).nav)
            for y in cfg["data"]["cash_yield_sensitivity"]}
    s0 = out["arms"]["S0_static"]
    for k in cfg["gate"]["applies_to"]:
        out["arms"][k]["gate"] = ee.gate(out["arms"][k], s0)
    out["forward_arm"] = ee.pick_forward({k: out["arms"][k] for k in cfg["gate"]["applies_to"]})
    return out


def log_trials(log: Path, out: Dict[str, Any], commit: str) -> None:
    for k, a in out["arms"].items():
        f = a["full"]
        psim.append_trial_log(log, stage="ETF_RISK_V1", market="TW+US", model=k,
                              change=f"one-shot frozen etf_risk_v1 ({commit[:7]}); calmar={f['calmar']:.3f}",
                              ic=np.nan, sharpe=np.nan, mdd=f["mdd"], cagr=f["cagr"],
                              t_nw=(a["vs_S0_monthly"] or {}).get("t_nw", np.nan),
                              turnover=a["annual_turnover"])


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot ETF risk-control research run (no orders)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trial-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    require_frozen(cfg)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"{args.out_dir} is not empty: this research run is one-shot")
    out = evaluate_arms(load_market(cfg), cfg)
    log_trials(args.trial_log, out, cfg["code_commit"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out["code_commit"] = cfg["code_commit"]
    (args.out_dir / "etf_risk_v1_summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"forward_arm": out["forward_arm"],
                      "gates": {k: out["arms"][k].get("gate") for k in cfg["gate"]["applies_to"]}}, indent=1))


if __name__ == "__main__":
    main()
