"""Assemble the Snowball v8 panel: PIT panel + chip / quality / macro-transmission / risk dimensions,
the v6 prior score and the pre-registered `v8_zero` composite.

Inputs come from (all kept OUTSIDE the git repo):
  build_tw_pit_panel.py --augment --save-daily   -> panel + tw_pit_daily_{close,raw_close,volume}
  tw_bulk_fetch.py --build-out <stem>            -> <stem>_{institutional,margin,foreign_hold}.parquet
  tw_mops_financials.py --out <quality>          -> PIT quality factors

Usage:
  python build_v8_panel.py --panel-dir ../../../../kaggle_dataset/v8 \
      --chip-stem ../../../../kaggle_dataset/v8/tw_chip --quality ../../../../kaggle_dataset/tw_quality_pit.parquet \
      --config ../../../configs/v8_frozen.json --out ../../../../kaggle_dataset/v8/snowball_tw_v8.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

import industry_detail as ind
import tw_mops_financials as tmf
import tw_pit_universe as tpu
import v8_composite as vc
import v8_features as vf
from train_walkforward import _baseline_prior_score

LOG = logging.getLogger("build_v8_panel")
MARKET_SYMBOLS = {"bench": "^TWII", "usd": "TWD=X", "us10y": "^TNX", "oil": "CL=F", "spx": "^GSPC"}
KEY = ["date", "ticker"]


def download_market_daily(start: str, end: str) -> pd.DataFrame:
    cols = {}
    for name, sym in MARKET_SYMBOLS.items():
        raw = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            raise RuntimeError(f"Failed to download {sym}")
        close = raw["Close"]
        cols[name] = (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).dropna()
    return pd.DataFrame(cols).sort_index()


def assemble(panel: pd.DataFrame, daily: Dict[str, pd.DataFrame], chip: Dict[str, pd.DataFrame],
             quality: pd.DataFrame, market: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pure assembly step (no I/O). Returns the panel with every v8 column added, row count unchanged."""
    base = panel.copy()
    base["date"] = pd.to_datetime(base["date"]).astype("datetime64[ns]")
    base["stock_id"] = base["ticker"].str.split(".").str[0]
    weeks = pd.DatetimeIndex(sorted(base["date"].unique()))
    id_to_ticker = dict(zip(base["stock_id"], base["ticker"]))
    close, volume = daily["close"], daily["volume"]
    traded_value = daily["raw_close"] * volume

    parts = [
        vf.chip_flow_features(chip["institutional"], volume, weeks, id_to_ticker),
        vf.chip_stock_features(chip["margin"], chip["foreign_hold"], volume, weeks, id_to_ticker),
        vf.risk_features(close, traded_value, market["bench"], weeks),
        vf.macro_features(vf.asof_weekly(close, weeks), vf.asof_weekly(market[["bench"]], weeks)["bench"],
                          vf.asof_weekly(market[["usd", "us10y", "oil", "spx"]], weeks)),
    ]
    out = base
    for part in parts:
        part = part.assign(date=pd.to_datetime(part["date"]).astype("datetime64[ns]"))
        out = out.merge(part, on=KEY, how="left")
    q = tmf.merge_asof_quality(out[["date", "stock_id"]].reset_index(), quality, max_staleness_days=200)
    out = out.join(q.set_index("index")[["q_gm_yoy", "q_roa_yoy", "q_asset_growth"]])
    out["pred_baseline_v6_prior"] = _baseline_prior_score(out)
    out = vc.add_v8_zero(out, cfg)
    if len(out) != len(panel):
        raise RuntimeError(f"Row count changed during assembly: {len(panel)} -> {len(out)}")
    return out


def coverage_by_year(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Share of non-missing values per v8 input per year (data sanity only; no returns involved)."""
    cols = [c for dim in cfg["dimensions"].values() for c in dim] + ["v8_zero"]
    return df.groupby(df["date"].dt.year)[cols].apply(lambda g: g.notna().mean()).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the Snowball v8 panel")
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--chip-stem", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stock-info", type=Path, default=None,
                        help="FinMind TaiwanStockInfo cache (parquet, outside the repo); fetched when missing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = vc.load_config(args.config)
    panel = pd.read_parquet(args.panel_dir / "snowball_tw_pit_t2.parquet")
    daily = {k: pd.read_parquet(args.panel_dir / f"tw_pit_daily_{k}.parquet") for k in ("close", "raw_close", "volume")}
    chip = {k: pd.read_parquet(args.chip_stem.with_name(f"{args.chip_stem.name}_{k}.parquet"))
            for k in ("institutional", "margin", "foreign_hold")}
    start = (daily["close"].index.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (daily["close"].index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    market = download_market_daily(start, end)
    out = assemble(panel, daily, chip, pd.read_parquet(args.quality), market, cfg)
    info_path = args.stock_info or args.panel_dir / "tw_stock_info.parquet"
    if not info_path.exists():
        tpu.fetch_finmind("TaiwanStockInfo").to_parquet(info_path, index=False)
    out = ind.attach_detailed_industry(out, ind.detailed_industry_map(pd.read_parquet(info_path)))
    out.to_parquet(args.out, index=False, compression="zstd")
    cov = coverage_by_year(out, cfg)
    cov.to_csv(args.out.with_name(args.out.stem + "_coverage.csv"))
    LOG.info("Saved %s shape=%s config=%s\n%s", args.out, out.shape, vc.config_hash(cfg)[:12], cov.to_string())


if __name__ == "__main__":
    main()
