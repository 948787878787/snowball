"""Tiny daily price cache for the core ETF tools (kept next to the ledger, outside git).

A symbol is downloaded at most once per calendar day unless `refresh=True`; this keeps weekly checks
fast and avoids Yahoo rate limits. The downloader is injected so tests never touch the network.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

Downloader = Callable[[str, str, str, str], pd.Series]


def _path(cache_dir: Path, symbol: str, field: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{symbol}_{field}")
    return Path(cache_dir) / f"{safe}.csv"


def cached_series(symbol: str, field: str, start: str, end: str, cache_dir: Optional[Path],
                  downloader: Downloader, refresh: bool = False) -> pd.Series:
    if cache_dir is None:
        return downloader(symbol, field, start, end)
    path = _path(cache_dir, symbol, field)
    fresh = path.exists() and datetime.fromtimestamp(path.stat().st_mtime).date() == date.today()
    if fresh and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return s.rename(symbol)
    s = downloader(symbol, field, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.rename(symbol).to_csv(path)
    return s


def yahoo_downloader(symbol: str, field: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No price data for {symbol}")
    s = raw[field]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).dropna()


__all__ = ["cached_series", "yahoo_downloader"]
