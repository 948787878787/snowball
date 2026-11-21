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


def run(oos: pd.DataFrame, score: str, end: Optional[str], neutralize: bool = True) -> pd.DataFrame:
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

    rows: Dict[str, dict] = {}
    for name, cfg in VARIANTS.items():
        res = cfs.simulate(df, score_col, "px_twd", cfg)
        rows[name] = {**cfs.summarize(res.values, res.contributions), "Trades": res.n_trades,
                      "Blocked_Sells": res.blocked_sells, "Realized_PnL": res.realized_pnl,
                      "Fees": res.fees_paid, "Delist_Events": res.delist_events,
                      "Holdings_End": len(res.final_positions)}
    etf = _next_day_value(_yahoo_series("0050.TW", start_s, end_s, "Adj Close"), dates)
    rows["bench_0050_dca"] = cfs.summarize(*cfs.dca_benchmark(etf, BASE.monthly_cash, BASE.fee_rate, BASE.min_fee))
    rows["bench_ew_universe_dca"] = cfs.summarize(*ew_dca_benchmark(df, fx, BASE.monthly_cash))
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    parser = argparse.ArgumentParser(description="User real-rule cash-flow simulation vs benchmarks")
    parser.add_argument("--oos", type=Path, required=True)
    parser.add_argument("--score", default="pred_v7_final")
    parser.add_argument("--end", default=None, help="Last signal date to include (e.g. 2025-11-20)")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()
    tbl = run(pd.read_parquet(args.oos), args.score, args.end)
    pd.set_option("display.width", 220)
    print(tbl.round(4).to_string())
    if args.out_csv is not None:
        tbl.to_csv(args.out_csv)


if __name__ == "__main__":
    main()
