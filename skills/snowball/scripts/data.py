"""Data loading with versioned, TTL-expiring, universe-fingerprinted on-disk cache.

Sources:
- Yahoo Finance (yfinance): daily OHLCV for TW stocks, ^TWII, US lead and macro tickers.
  Downloads with `auto_adjust=False` so we retain:
  - `close`: dividend/split-adjusted close (`Adj Close`) for total-return calculations.
  - `raw_close`: unadjusted nominal close (`Close`) for point-in-time affordability filters
    (avoiding look-ahead from future dividends).
  - `open`: dividend/split-adjusted open (`Open * AdjClose / Close`) for realistic next-day
    open execution (eliminating same-bar Friday-close fill bias).
- FinMind open API: monthly revenue, PER/PBR/dividend yield, institutional buy/sell.

Cache Invalidation Guarantees:
- Every cached artifact writes an atomic `.pkl` + `.meta.json` sidecar containing:
  `schema_version`, `universe_hash` (SHA-256 of sorted stock IDs + START_DATE), `created_at`, `ttl_seconds`.
- Automatically invalidates and rebuilds on: TTL expiration, `CACHE_SCHEMA_VERSION` bump,
  universe membership change, missing/corrupt metadata (legacy caches), or `SNOWBALL_FORCE_REFRESH=1`.
- Default TTLs: 24h for daily prices/valuation/institutional/panels (`SNOWBALL_CACHE_TTL_HOURS`),
  7 days for monthly revenue (`SNOWBALL_REV_CACHE_TTL_HOURS`).

Point-in-time rules applied here:
- Monthly revenue is treated as available on the 11th of the month after the revenue month
  (legal deadline is the 10th), never earlier.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import requests
import yfinance as yf

from universe import MACRO_TICKERS, MARKET_INDEX, US_LEAD_TICKERS, all_stock_ids

LOG = logging.getLogger(__name__)
CACHE_DIR = Path(os.environ.get("SNOWBALL_CACHE_DIR", str(Path(__file__).parent / "cache")))
CACHE_SCHEMA_VERSION = "v3"
DEFAULT_DAILY_TTL_SECONDS = float(os.environ.get("SNOWBALL_CACHE_TTL_HOURS", "24")) * 3600.0
DEFAULT_MONTHLY_TTL_SECONDS = float(os.environ.get("SNOWBALL_REV_CACHE_TTL_HOURS", "168")) * 3600.0

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
START_DATE = "2017-01-01"
REQUEST_PAUSE_SEC = 1.2
REVENUE_PUBLISH_LAG_DAYS = 10


def universe_fingerprint() -> str:
    """Stable short hash of the current universe membership and start date."""
    payload = f"{START_DATE}|" + ",".join(sorted(all_stock_ids()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _cache_dir() -> Path:
    env_dir = os.environ.get("SNOWBALL_CACHE_DIR", "").strip()
    target = Path(env_dir) if env_dir else CACHE_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _cache_path(name: str, cache_dir: Optional[Path] = None) -> Path:
    base = cache_dir if cache_dir is not None else _cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.pkl"


def _meta_path(name: str, cache_dir: Optional[Path] = None) -> Path:
    base = cache_dir if cache_dir is not None else _cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.meta.json"


def _read_meta(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        return meta if isinstance(meta, dict) else None
    except (OSError, ValueError):
        return None


def _is_env_force_refresh() -> bool:
    return os.environ.get("SNOWBALL_FORCE_REFRESH", "").strip().lower() in {"1", "true", "yes"}


def cached_load(
    name: str,
    loader: Callable[[], Any],
    *,
    ttl_seconds: float = DEFAULT_DAILY_TTL_SECONDS,
    schema_version: str = CACHE_SCHEMA_VERSION,
    fingerprint: Optional[str] = None,
    force_refresh: bool = False,
    allow_stale_on_error: bool = True,
    cache_dir: Optional[Path] = None,
    now_fn: Callable[[], float] = time.time,
) -> Any:
    """Load a cached Python/pandas object with strict TTL, schema-version, and universe-hash validation.

    - Never serves a cache with mismatched `schema_version`, mismatched `fingerprint`, or missing metadata.
    - If a schema+universe-valid cache is merely expired by TTL and `loader()` fails due to a transient
      network/API error, optionally falls back to the stale cache when `allow_stale_on_error=True`.
    """
    pkl_path = _cache_path(name, cache_dir=cache_dir)
    meta_path = _meta_path(name, cache_dir=cache_dir)
    expected_fp = fingerprint if fingerprint is not None else universe_fingerprint()
    now = now_fn()
    refresh = force_refresh or _is_env_force_refresh()

    meta = _read_meta(meta_path)
    compatible = (
        pkl_path.exists()
        and meta is not None
        and meta.get("schema_version") == schema_version
        and meta.get("universe_hash") == expected_fp
    )

    if compatible and not refresh:
        age = now - float(meta.get("created_at", 0.0))
        effective_ttl = min(float(ttl_seconds), float(meta.get("ttl_seconds", ttl_seconds)))
        if 0.0 <= age <= effective_ttl:
            return pd.read_pickle(pkl_path)
        LOG.info("Cache '%s' expired (age=%.0fs > ttl=%.0fs); refreshing.", name, age, effective_ttl)
    elif pkl_path.exists() and not compatible:
        LOG.info("Cache '%s' invalidated (schema/universe mismatch or legacy format); rebuilding.", name)

    try:
        obj = loader()
    except Exception as exc:
        if compatible and allow_stale_on_error and not refresh:
            LOG.warning("Live refresh failed for '%s' (%s); falling back to expired compatible cache.", name, exc)
            return pd.read_pickle(pkl_path)
        raise

    # Atomic write for pickle + metadata sidecar
    tmp_pkl = pkl_path.with_suffix(".pkl.tmp")
    tmp_meta = meta_path.with_suffix(".meta.json.tmp")
    pd.to_pickle(obj, tmp_pkl)
    os.replace(tmp_pkl, pkl_path)
    tmp_meta.write_text(
        json.dumps(
            {
                "name": name,
                "schema_version": schema_version,
                "universe_hash": expected_fp,
                "created_at": now_fn(),
                "ttl_seconds": float(ttl_seconds),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_meta, meta_path)
    return obj


def clear_cache(name: Optional[str] = None, cache_dir: Optional[Path] = None) -> int:
    """Remove one named cache entry (or all cache files if `name=None`). Returns files removed."""
    base = cache_dir if cache_dir is not None else _cache_dir()
    if not base.exists():
        return 0
    removed = 0
    targets = (
        [_cache_path(name, base), _meta_path(name, base)]
        if name is not None
        else [*base.glob("*.pkl"), *base.glob("*.meta.json")]
    )
    for p in targets:
        if p.exists():
            p.unlink()
            removed += 1
    return removed


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


def load_prices(force_refresh: bool = False, ttl_seconds: float = DEFAULT_DAILY_TTL_SECONDS) -> dict:
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
    return cached_load("prices_v2", loader, ttl_seconds=ttl_seconds, force_refresh=force_refresh)["payload"].iloc[0]


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


def _load_finmind_all(
    dataset: str,
    ttl_seconds: float = DEFAULT_DAILY_TTL_SECONDS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    def loader():
        frames = []
        for sid in all_stock_ids():
            frames.append(_finmind(dataset, sid))
            time.sleep(REQUEST_PAUSE_SEC)
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return cached_load(dataset, loader, ttl_seconds=ttl_seconds, force_refresh=force_refresh)


def load_revenue(force_refresh: bool = False, ttl_seconds: float = DEFAULT_MONTHLY_TTL_SECONDS) -> pd.DataFrame:
    """Monthly revenue with a point-in-time 'available' date."""
    df = _load_finmind_all("TaiwanStockMonthRevenue", ttl_seconds=ttl_seconds, force_refresh=force_refresh).copy()
    df = df.assign(
        date=pd.to_datetime(df["date"]),
        available=pd.to_datetime(df["date"]) + pd.Timedelta(days=REVENUE_PUBLISH_LAG_DAYS),
    )
    return df[["stock_id", "revenue_year", "revenue_month", "revenue", "available"]]


def load_valuation(force_refresh: bool = False, ttl_seconds: float = DEFAULT_DAILY_TTL_SECONDS) -> pd.DataFrame:
    df = _load_finmind_all("TaiwanStockPER", ttl_seconds=ttl_seconds, force_refresh=force_refresh).copy()
    return df.assign(date=pd.to_datetime(df["date"]))


def load_institutional(force_refresh: bool = False, ttl_seconds: float = DEFAULT_DAILY_TTL_SECONDS) -> pd.DataFrame:
    df = _load_finmind_all("TaiwanStockInstitutionalInvestorsBuySell", ttl_seconds=ttl_seconds, force_refresh=force_refresh).copy()
    return df.assign(date=pd.to_datetime(df["date"]), net=df["buy"] - df["sell"])
