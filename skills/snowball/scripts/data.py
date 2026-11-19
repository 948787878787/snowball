"""Data loading with on-disk cache.

Sources:
- Yahoo Finance (yfinance): daily OHLCV for TW stocks, ^TWII, US lead and macro tickers.
  Downloads with `auto_adjust=False` so we retain:
  - `close`: dividend/split-adjusted close (`Adj Close`) for total-return calculations.
  - `raw_close`: unadjusted nominal close (`Close`) for point-in-time affordability filters
    (avoiding look-ahead from future dividends).
  - `open`: dividend/split-adjusted open (`Open * AdjClose / Close`) for realistic next-day
    open execution (eliminating same-bar Friday-close fill bias).
- FinMind open API: monthly revenue, PER/PBR/dividend yield, institutional buy/sell.

Point-in-time rules applied here:
- Monthly revenue is treated as available on the 11th of the month after the revenue month
  (legal deadline is the 10th), never earlier.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from universe import MACRO_TICKERS, MARKET_INDEX, US_LEAD_TICKERS, all_stock_ids

LOG = logging.getLogger(__name__)
CACHE_DIR = Path(os.environ.get("SNOWBALL_CACHE_DIR", str(Path(__file__).parent / "cache")))
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
START_DATE = "2017-01-01"
REQUEST_PAUSE_SEC = 1.2
REVENUE_PUBLISH_LAG_DAYS = 10


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def _cached(name: str, loader):
    # Pickle is safe here: cache files are written only by this script into a local folder.
    path = _cache_path(name)
    if path.exists():
        return pd.read_pickle(path)
    df = loader()
    df.to_pickle(path)
    return df


def _download_yahoo(tickers: list, auto_adjust: bool = True) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers, start=START_DATE, progress=False, auto_adjust=auto_adjust, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return raw


def _resolve_tw_symbols(stock_ids: tuple) -> dict:
    """Map stock id -> Yahoo symbol (.TW for TWSE, .TWO for TPEx)."""
    first = _download_yahoo([f"{s}.TW" for s in stock_ids], auto_adjust=True)
    if first.empty or "Close" not in first:
        raise RuntimeError("Failed to download Yahoo Finance prices. Check network connectivity.")
    closes = first["Close"]
    mapping = {}
    missing = []
    for sid in stock_ids:
        col = f"{sid}.TW"
        if col in closes and closes[col].notna().sum() > 200:
            mapping[sid] = col
        else:
            missing.append(sid)
    if missing:
        second_raw = _download_yahoo([f"{s}.TWO" for s in missing], auto_adjust=True)
        second = second_raw["Close"] if (not second_raw.empty and "Close" in second_raw) else pd.DataFrame()
        for sid in missing:
            col = f"{sid}.TWO"
            if col in second and second[col].notna().sum() > 200:
                mapping[sid] = col
            else:
                LOG.warning("No Yahoo data for %s", sid)
    if not mapping:
        raise RuntimeError("No valid Taiwan stock symbols resolved from Yahoo Finance.")
    return mapping


def load_prices() -> dict:
    """Return dict with 'close' (adjusted), 'raw_close' (nominal), 'open' (adjusted),
    'volume', 'index', 'us', and 'macro' frames."""
    def loader():
        ids = all_stock_ids()
        mapping = _resolve_tw_symbols(ids)
        raw = _download_yahoo(list(mapping.values()), auto_adjust=False)
        inverse = {v: k for k, v in mapping.items()}
        raw_close = raw["Close"].rename(columns=inverse)
        adj_close = raw["Adj Close"].rename(columns=inverse) if "Adj Close" in raw else raw_close.copy()
        adj_ratio = adj_close / raw_close.replace(0.0, pd.NA)
        adj_open = (raw["Open"].rename(columns=inverse) * adj_ratio) if "Open" in raw else adj_close.copy()
        volume = raw["Volume"].rename(columns=inverse)
        ext = _download_yahoo([MARKET_INDEX, *US_LEAD_TICKERS, *MACRO_TICKERS], auto_adjust=True)["Close"]
        return pd.DataFrame({"payload": [{
            "close": adj_close,
            "raw_close": raw_close,
            "open": adj_open,
            "volume": volume,
            "index": ext[MARKET_INDEX],
            "us": ext[list(US_LEAD_TICKERS)],
            "macro": ext[list(MACRO_TICKERS)],
        }]})
    return _cached("prices_v2", loader)["payload"].iloc[0]


def _finmind(dataset: str, stock_id: str) -> pd.DataFrame:
    params = {"dataset": dataset, "data_id": stock_id, "start_date": START_DATE}
    for attempt in range(3):
        try:
            resp = requests.get(FINMIND_URL, params=params, timeout=60)
            payload = resp.json()
            if resp.status_code == 200 and payload.get("msg") == "success":
                return pd.DataFrame(payload.get("data", []))
            LOG.warning("FinMind %s %s: %s %s", dataset, stock_id, resp.status_code, payload.get("msg"))
        except (requests.RequestException, ValueError) as exc:
            LOG.warning("FinMind %s %s attempt %d failed: %s", dataset, stock_id, attempt, exc)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"FinMind fetch failed: {dataset} {stock_id}")


def _load_finmind_all(dataset: str) -> pd.DataFrame:
    def loader():
        frames = []
        for sid in all_stock_ids():
            frames.append(_finmind(dataset, sid))
            time.sleep(REQUEST_PAUSE_SEC)
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return _cached(dataset, loader)


def load_revenue() -> pd.DataFrame:
    """Monthly revenue with a point-in-time 'available' date."""
    df = _load_finmind_all("TaiwanStockMonthRevenue").copy()
    df = df.assign(
        date=pd.to_datetime(df["date"]),
        available=pd.to_datetime(df["date"]) + pd.Timedelta(days=REVENUE_PUBLISH_LAG_DAYS),
    )
    return df[["stock_id", "revenue_year", "revenue_month", "revenue", "available"]]


def load_valuation() -> pd.DataFrame:
    df = _load_finmind_all("TaiwanStockPER").copy()
    return df.assign(date=pd.to_datetime(df["date"]))


def load_institutional() -> pd.DataFrame:
    df = _load_finmind_all("TaiwanStockInstitutionalInvestorsBuySell").copy()
    return df.assign(date=pd.to_datetime(df["date"]), net=df["buy"] - df["sell"])
