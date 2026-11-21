"""Build the survivorship-free point-in-time TW training panel for Snowball v7 T2.

Pipeline:
  1. Candidates: every listed TWSE/TPEx common stock (FinMind `TaiwanStockInfo`) + every common stock
     delisted since the panel start (FinMind `TaiwanStockDelisting`).
  2. Prices: Yahoo (adjusted) for listed names; FinMind (unadjusted) for delisted names Yahoo lacks.
  3. Monthly PIT membership mask (top-N by 20-day traded value, effective after month-end).
  4. Features / labels via `build_training_panel._build_single_market_panel` with delisting settlement.
  5. Mask applied, target ranks recomputed within the member cross-section.
  6. Optional Notebook A augmentations: TWSE/TPEx PIT valuation, MOPS monthly revenue, 52w high.

Outputs go to `--out-dir` (keep it OUTSIDE the git repo; data files must not be committed).

Usage:
  python build_tw_pit_panel.py --out-dir ../../../../kaggle_dataset --period 8y --top-n 1000 --augment
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd
import yfinance as yf

import build_training_panel as btp
import portfolio_sim as psim
import tw_pit_universe as tpu
from global_ranker import convert_prices_to_usd
from global_universe import get_market_groups
from macro_micro import download_macro_series

LOG = logging.getLogger("build_tw_pit_panel")
BENCHMARK = "^TWII"
_PERIOD_YEARS = {"5y": 5, "8y": 8, "10y": 10, "15y": 15}


def rerank_targets(panel: pd.DataFrame, horizons: Sequence[int] = (4, 12)) -> pd.DataFrame:
    """Recompute target percentile ranks within the (date, market) member cross-section."""
    out = panel.copy()
    for h in horizons:
        col = f"fwd_vol_adj_{h}w"
        if col not in out.columns:
            continue
        mkt_rank = out.groupby(["date", "market"])[col].rank(pct=True).astype("float32")
        out[f"target_rank_{h}w"] = mkt_rank
        out[f"target_rank_mkt_{h}w"] = mkt_rank
        out[f"target_rank_global_{h}w"] = out.groupby("date")[col].rank(pct=True).astype("float32")
    return out


def merge_groups(base: Mapping[str, Tuple[str, ...]], extra: pd.DataFrame) -> Dict[str, Tuple[str, ...]]:
    """Keep curated industry for known tickers; add new tickers using their FinMind-mapped group."""
    known = {t for mems in base.values() for t in mems}
    merged = {g: list(mems) for g, mems in base.items()}
    for row in extra.itertuples(index=False):
        if row.ticker in known:
            continue
        merged.setdefault(row.group, []).append(row.ticker)
        known.add(row.ticker)
    return {g: tuple(mems) for g, mems in merged.items()}


def combine_price_sources(yahoo: pd.DataFrame, finmind: pd.DataFrame) -> pd.DataFrame:
    """Union of dates; Yahoo columns win, FinMind only fills tickers Yahoo does not cover."""
    extra_cols = [c for c in finmind.columns if c not in yahoo.columns]
    idx = yahoo.index.union(finmind.index)
    out = pd.concat([yahoo.reindex(idx), finmind[extra_cols].reindex(idx)], axis=1)
    return out.sort_index()


def _start_date(period: str) -> str:
    years = _PERIOD_YEARS.get(period)
    if years is None:
        raise ValueError(f"Unsupported --period {period}; choose from {sorted(_PERIOD_YEARS)}")
    return (pd.Timestamp.today().normalize() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")


def _download_benchmark(period: str) -> pd.Series:
    raw = yf.download(BENCHMARK, period=period, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"Failed to download benchmark {BENCHMARK}")
    close = raw["Close"]
    return (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).dropna()


def fetch_universe_prices(period: str, batch_size: int):
    """Return candidate metadata plus TWD daily (adj close, raw close, open, volume) and delisting flags."""
    start = _start_date(period)
    listed = tpu.select_common_stocks(tpu.fetch_finmind("TaiwanStockInfo"))
    delisted = tpu.filter_delistings(tpu.fetch_finmind("TaiwanStockDelisting"), start=start)
    ticker_map = dict(zip(listed["stock_id"], listed["ticker"]))
    live = listed.loc[~listed["stock_id"].isin(delisted["stock_id"])]
    LOG.info("Candidates: %d listed, %d delisted since %s", len(live), len(delisted), start)

    y_close, y_raw, y_open, y_vol = btp._download_batched_ohlcv(list(live["ticker"]), period=period, batch_size=batch_size)
    f_close, f_open, f_vol, stale = tpu.fetch_delisted_ohlcv(delisted, ticker_map, start=start)
    LOG.info("Yahoo covered %d/%d listed; FinMind covered %d/%d delisted", y_close.shape[1], len(live), f_close.shape[1], len(delisted))

    prices = {
        "close": combine_price_sources(y_close, f_close),
        "raw_close": combine_price_sources(y_raw, f_close),
        "open": combine_price_sources(y_open, f_open),
        "volume": combine_price_sources(y_vol, f_vol),
    }
    group_of = dict(zip(listed["ticker"], listed["group"]))
    delisted_groups = pd.DataFrame({"ticker": list(f_close.columns)})
    delisted_groups["group"] = delisted_groups["ticker"].map(group_of).fillna("other")
    meta = pd.concat([live[["ticker", "group"]], delisted_groups], ignore_index=True)
    return meta, prices, stale


def build_panel(period: str, top_n: int, batch_size: int, horizons: Sequence[int] = (4, 12)):
    """Build the masked PIT panel; returns (panel, mask, mask_summary)."""
    meta, px, stale = fetch_universe_prices(period, batch_size)
    trading_days = px["close"].index
    weeks = pd.date_range(trading_days.min(), trading_days.max(), freq="W-FRI")
    mask = tpu.build_monthly_membership_mask(px["raw_close"], px["volume"], weeks, top_n=top_n)
    summary = tpu.summarize_mask(mask, stale.keys())

    close_usd, open_usd = convert_prices_to_usd(px["close"], px["open"], period=period)
    raw_usd, _ = convert_prices_to_usd(px["raw_close"], px["raw_close"].copy(), period=period)
    bench = _download_benchmark(period).to_frame(name=BENCHMARK)
    bench_usd = convert_prices_to_usd(bench, bench.copy(), period=period)[0][BENCHMARK]
    groups = merge_groups(get_market_groups("TW"), meta)

    frame = btp._build_single_market_panel(
        market="TW", close_d=close_usd, open_d=open_usd, vol_d=px["volume"], bench_d=bench_usd,
        groups=groups, macro_d=download_macro_series(period=period), horizons=horizons,
        raw_close_d=raw_usd, delisted_stale=stale,
    )
    panel = rerank_targets(psim._apply_membership_mask(frame, mask), horizons)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True), mask, summary


def augment_like_notebook_a(panel: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Same Stage 4 augmentations as Notebook A Cell 2 (PIT valuation, MOPS revenue, 52w high)."""
    import fetch_pit_fundamentals as fpf
    import tw_monthly_rev as tmr

    tickers = sorted(panel["ticker"].unique().tolist())
    start_s = (panel["date"].min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end_s = panel["date"].max().strftime("%Y-%m-%d")
    pit_tw = fpf.fetch_tw_exchange_pit(tickers, cache_dir=cache_dir, start_date=start_s, end_date=end_s)
    if not pit_tw.empty:
        panel = fpf.merge_pit_into_panel(panel, pit_tw, max_staleness_days=200)
    rev_raw = tmr.fetch_mops_monthly_revenue(
        int(panel["date"].min().year) - 1, int(panel["date"].max().year),
        cache_dir=cache_dir, tickers_filter=tickers, max_workers=8,
    )
    if not rev_raw.empty:
        panel = tmr.merge_tw_monthly_rev_into_panel(panel, tmr.compute_tw_monthly_rev_features(rev_raw), max_staleness_days=65)
    return tmr.add_52w_high_proximity(panel, price_col="close_usd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball v7 T2 survivorship-free TW PIT panel")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory (outside the git repo)")
    parser.add_argument("--period", default="8y", choices=sorted(_PERIOD_YEARS))
    parser.add_argument("--top-n", type=int, default=tpu.DEFAULT_TOP_N)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--augment", action="store_true", help="Add PIT valuation, MOPS revenue and 52w high")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel, mask, summary = build_panel(args.period, args.top_n, args.batch_size)
    mask.to_parquet(args.out_dir / "tw_pit_membership_mask.parquet")
    summary.to_csv(args.out_dir / "tw_pit_mask_summary.csv")
    LOG.info("Mask summary by year:\n%s", summary.to_string())
    if args.augment:
        panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
        panel = augment_like_notebook_a(panel, args.cache_dir or args.out_dir / "pit_cache")
    out_path = args.out_dir / "snowball_tw_pit_t2.parquet"
    panel.to_parquet(out_path, index=False, compression="zstd")
    LOG.info("Saved %s shape=%s tickers=%d", out_path, panel.shape, panel["ticker"].nunique())


if __name__ == "__main__":
    main()
