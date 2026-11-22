"""Snowball v7 T2 report: compare the zero-parameter prior IC on the old (today's top-1,000) universe
versus the survivorship-free point-in-time universe, by calendar year.

The zero-parameter prior has no trained weights, so any IC change between universes is caused by the
universe definition itself (selection / survivorship bias), not by model fitting.

Usage:
  python t2_universe_report.py --old-panel OLD.parquet --new-panel NEW.parquet [--out-csv report.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

import train_walkforward as twf


def yearly_prior_ic(panel: pd.DataFrame, horizons: Sequence[int] = (4, 12), market: str = "TW") -> pd.DataFrame:
    """Mean weekly cross-sectional Spearman IC of the zero-parameter prior, per calendar year."""
    df = panel.loc[panel["market"] == market].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["prior_score"] = twf._baseline_prior_score(df)
    cols = {}
    for h in horizons:
        ic = twf._compute_daily_ic_series(df, "prior_score", f"target_rank_{h}w")
        cols[f"ic_{h}w"] = ic.groupby(ic.index.year).mean()
        if h == horizons[0]:
            cols["n_weeks"] = ic.groupby(ic.index.year).size()
    return pd.DataFrame(cols)


def compare_universes(old: pd.DataFrame, new: pd.DataFrame, horizons: Sequence[int] = (4, 12)) -> pd.DataFrame:
    """Side-by-side yearly prior IC (old vs PIT universe) plus average member counts."""
    a = yearly_prior_ic(old, horizons).add_prefix("old_")
    b = yearly_prior_ic(new, horizons).add_prefix("pit_")
    out = a.join(b, how="outer")
    for name, df in (("old", old), ("pit", new)):
        d = pd.to_datetime(df.loc[df["market"] == "TW", "date"])
        out[f"{name}_avg_stocks"] = d.groupby(d.dt.year).size() / d.groupby(d.dt.year).nunique()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="T2 old vs PIT universe zero-parameter prior IC by year")
    parser.add_argument("--old-panel", type=Path, required=True)
    parser.add_argument("--new-panel", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()
    tbl = compare_universes(pd.read_parquet(args.old_panel), pd.read_parquet(args.new_panel))
    pd.set_option("display.width", 200)
    print(tbl.round(4).to_string())
    if args.out_csv is not None:
        tbl.to_csv(args.out_csv)


if __name__ == "__main__":
    main()
