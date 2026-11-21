"""Run the user's real-rule cash-flow simulation on Snowball OOS predictions and compare it with
same-cash-flow benchmarks (0050 DCA, equal-weight universe DCA).

Variants (fixed before looking at results, not tuned):
  real_rules       : buy top-2% (Top-20) name each week cash allows, sell below 60th pct only if profitable
  allow_loss_sells : same, but losing positions in the sell zone are sold too
  never_sell       : buy-only, holdings never sold

Usage:
  python run_cashflow_sim.py --oos OOS.parquet --score pred_v7_final [--end 2025-11-20] [--out-csv out.csv]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

import cashflow_sim as cfs
import portfolio_sim as psim
import stage5_validation as s5

BASE = cfs.CashflowConfig()
VARIANTS = {
    "real_rules": BASE,
    "allow_loss_sells": replace(BASE, allow_loss_sells=True),
    "never_sell": replace(BASE, sell_pct=0.0),
}


def _yahoo_series(symbol: str, start: str, end: str, field: str) -> pd.Series:
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    s = raw[field]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).dropna()


def _next_day_value(daily: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Value on the first trading day strictly after each signal date (execution-day price)."""
    pos = daily.index.searchsorted(dates, side="right")
    vals = [daily.iloc[p] if p < len(daily) else float("nan") for p in pos]
    return pd.Series(vals, index=dates)


def ew_dca_benchmark(df: pd.DataFrame, fx: pd.Series, monthly_cash: float) -> tuple:
    """Equal-weight universe DCA (not investable; isolates stock-selection value). Returns in TWD."""
    ew_usd = df.groupby("date")["fwd_ret_1w_open"].mean()
    fx_ret = fx.shift(-1) / fx - 1.0
    ew_twd = ((1.0 + ew_usd) * (1.0 + fx_ret) - 1.0).fillna(0.0)
    vals, contribs, value, prev = {}, {}, 0.0, None
    for d in ew_usd.index:
        month = (d.year, d.month)
        contribs[d] = monthly_cash if month != prev else 0.0
        prev = month
        value += contribs[d]
        vals[d] = value
        value *= 1.0 + float(ew_twd.get(d, 0.0))
    return pd.Series(vals), pd.Series(contribs)


def run(
    oos: pd.DataFrame, score: str, end: Optional[str], neutralize: bool = True,
    price_wide_usd: Optional[pd.DataFrame] = None, delist_haircut: float = cfs.DEFAULT_DELIST_HAIRCUT,
) -> pd.DataFrame:
    df = oos.loc[oos["market"] == "TW"].copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    if end is not None:
        df = df.loc[df["date"] <= pd.Timestamp(end)]
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    start_s, end_s = (dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), (dates.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    fx = _next_day_value(_yahoo_series("TWD=X", start_s, end_s, "Close"), dates)
    df["px_twd"] = df["next_open_usd"] * df["date"].map(fx)
    score_col = score
    if neutralize:
        df = psim.neutralize_score_panel(df, score, out_col=f"{score}_neut")
        score_col = f"{score}_neut"

    fallback = None
    if price_wide_usd is not None:
        wide = price_wide_usd.copy()
        wide.index = pd.to_datetime(wide.index).astype("datetime64[ns]")
        wide = wide.reindex(dates)
        fallback = wide.mul(fx.reindex(dates), axis=0)

    etf = _next_day_value(_yahoo_series("0050.TW", start_s, end_s, "Adj Close"), dates)
    etf_v, etf_c = cfs.dca_benchmark(etf, BASE.monthly_cash, BASE.fee_rate, BASE.min_fee)
    ew_v, ew_c = ew_dca_benchmark(df, fx, BASE.monthly_cash)
    bench_twr = {"EW": cfs.weekly_twr(ew_v, ew_c), "0050": cfs.weekly_twr(etf_v, etf_c)}

    rows: Dict[str, dict] = {}
    for name, cfg in VARIANTS.items():
        res = cfs.simulate(df, score_col, "px_twd", cfg, delist_haircut=delist_haircut, price_fallback=fallback)
        twr = cfs.weekly_twr(res.values, res.contributions)
        sig = {f"t_NW_vs_{b}": s5.compute_excess_metrics(twr, bw, lags=4)["Excess_t_NW"] for b, bw in bench_twr.items()}
        rows[name] = {**cfs.summarize(res.values, res.contributions), **sig, "Trades": res.n_trades,
                      "Blocked_Sells": res.blocked_sells, "Realized_PnL": res.realized_pnl,
                      "Fees": res.fees_paid, "Delist_Events": res.delist_events,
                      "Holdings_End": len(res.final_positions)}
    rows["bench_0050_dca"] = {**cfs.summarize(etf_v, etf_c),
                              "t_NW_vs_EW": s5.compute_excess_metrics(bench_twr["0050"], bench_twr["EW"], lags=4)["Excess_t_NW"]}
    rows["bench_ew_universe_dca"] = cfs.summarize(ew_v, ew_c)
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    parser = argparse.ArgumentParser(description="User real-rule cash-flow simulation vs benchmarks")
    parser.add_argument("--oos", type=Path, required=True)
    parser.add_argument("--score", default="pred_v7_final")
    parser.add_argument("--end", default=None, help="Last signal date to include (e.g. 2025-11-20)")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--price-wide", type=Path, default=None,
                        help="PIT: unmasked weekly next-open USD prices (tw_pit_next_open_usd_wide.parquet)")
    parser.add_argument("--delist-haircut", type=float, default=cfs.DEFAULT_DELIST_HAIRCUT,
                        help="Use 0.0 for PIT panels, whose final prices already include the stale haircut")
    args = parser.parse_args()
    wide = pd.read_parquet(args.price_wide) if args.price_wide is not None else None
    tbl = run(pd.read_parquet(args.oos), args.score, args.end, price_wide_usd=wide, delist_haircut=args.delist_haircut)
    pd.set_option("display.width", 220)
    print(tbl.round(4).to_string())
    if args.out_csv is not None:
        tbl.to_csv(args.out_csv)


if __name__ == "__main__":
    main()
