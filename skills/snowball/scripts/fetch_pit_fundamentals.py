"""Point-in-Time (PIT) Historical Fundamentals Fetcher for US (SEC EDGAR XBRL) & TW (TWSE/TPEx).

Strictly enforces Stage 1 / P0-6 rules:
- Multi-tag XBRL union & 10-K Q4 TTM derivation via `sec_xbrl.parse_sec_company_facts`.
- 200-day staleness expiration (`tolerance=pd.Timedelta(days=200)`) and 270-day period-age cutoff.
- Split-immune US E/P: Priority 1 uses `ttm_net_income / (shares_out * raw_close_usd)` against
  unadjusted nominal `raw_close_usd`; Priority 2 uses split-adjusted `ttm_eps`.
- SEC User-Agent reads from `SNOWBALL_SEC_USER_AGENT` environment variable when provided.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_training_panel import load_dataframe, save_dataframe
from sec_xbrl import parse_sec_company_facts  # noqa: F401 (re-exported for callers & tests)

DEFAULT_MAX_STALENESS_DAYS = 200
DEFAULT_MAX_PERIOD_AGE_DAYS = 270

TW_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SnowballQuantResearch/1.0)",
    "Accept": "application/json, text/plain, */*",
}


def _sec_headers() -> Dict[str, str]:
    ua = os.environ.get(
        "SNOWBALL_SEC_USER_AGENT",
        "SnowballQuantResearch/1.0 (contact@snowball-quant.org)",
    ).strip()
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


class _RateLimiter:
    """Thread-safe interval limiter (keeps SEC requests strictly <= 7 req/sec)."""

    def __init__(self, min_interval_sec: float = 0.14):
        self.min_interval = float(min_interval_sec)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_s = self.min_interval - (now - self._last)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last = time.monotonic()


def _parse_float(val: object) -> float:
    if val is None:
        return np.nan
    s = str(val).strip().replace(",", "").replace("--", "")
    if not s or s in ("-", "N/A", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _load_sec_cik_map(sec_cache: Path, headers: Dict[str, str], offline_cache_only: bool) -> Dict[str, str]:
    cik_map_file = sec_cache / "company_tickers.json"
    if cik_map_file.exists():
        cik_raw = json.loads(cik_map_file.read_text(encoding="utf-8"))
    elif offline_cache_only:
        raise FileNotFoundError(f"SEC CIK cache missing in offline mode: {cik_map_file}")
    else:
        resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=20)
        resp.raise_for_status()
        cik_raw = resp.json()
        cik_map_file.write_text(json.dumps(cik_raw), encoding="utf-8")

    ticker_to_cik: Dict[str, str] = {}
    for item in cik_raw.values():
        t = str(item.get("ticker", "")).upper().strip()
        cik = f"{int(item['cik_str']):010d}"
        if t:
            ticker_to_cik[t] = cik
            ticker_to_cik[t.replace("-", ".")] = cik
            ticker_to_cik[t.replace(".", "-")] = cik
    return ticker_to_cik


def _fetch_one_sec_ticker(
    sym: str,
    cik: str,
    sec_cache: Path,
    min_year: int,
    offline_cache_only: bool,
    limiter: _RateLimiter,
    session: requests.Session,
) -> pd.DataFrame:
    cfile = sec_cache / f"CIK{cik}.json"
    if cfile.exists():
        try:
            return parse_sec_company_facts(sym, json.loads(cfile.read_text(encoding="utf-8")), min_year=min_year)
        except Exception:
            return pd.DataFrame()
    if offline_cache_only:
        return pd.DataFrame()
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    for attempt in range(3):
        limiter.wait()
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                cfile.write_bytes(r.content)
                return parse_sec_company_facts(sym, r.json(), min_year=min_year)
            if r.status_code == 404:
                return pd.DataFrame()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.0 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()


def fetch_us_sec_pit(
    us_tickers: Sequence[str],
    cache_dir: Path,
    min_year: int = 2018,
    max_workers: int = 6,
    offline_cache_only: bool = False,
) -> pd.DataFrame:
    """Fetch (or re-parse from local cache) Point-in-Time XBRL fundamentals for US tickers."""
    sec_cache = cache_dir / "sec"
    sec_cache.mkdir(parents=True, exist_ok=True)
    headers = _sec_headers()
    ticker_to_cik = _load_sec_cik_map(sec_cache, headers, offline_cache_only)

    targets: List[Tuple[str, str]] = []
    for sym in dict.fromkeys(us_tickers):
        k = sym.upper().strip()
        cik = ticker_to_cik.get(k) or ticker_to_cik.get(k.replace("-", ".")) or ticker_to_cik.get(k.replace(".", "-"))
        if cik:
            targets.append((sym, cik))

    limiter = _RateLimiter(min_interval_sec=0.14)
    session = requests.Session()
    session.headers.update(headers)
    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_fetch_one_sec_ticker, sym, cik, sec_cache, min_year, offline_cache_only, limiter, session): sym
            for sym, cik in targets
        }
        for fut in as_completed(futs):
            try:
                df_one = fut.result()
                if not df_one.empty:
                    frames.append(df_one)
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_tw_rows(
    records: Sequence[Sequence[object]],
    fields: Sequence[str],
    suffix: str,
    snap_date: str,
    allowed_codes: set,
) -> pd.DataFrame:
    idx_pe = next((i for i, f in enumerate(fields) if "本益比" in f), -1)
    idx_pb = next((i for i, f in enumerate(fields) if "淨值比" in f), -1)
    idx_yld = next((i for i, f in enumerate(fields) if "殖利率" in f), -1)
    avail = (pd.Timestamp(snap_date) + pd.Timedelta(days=1)).date().isoformat()
    max_i = max(idx_pe, idx_pb, idx_yld, 0)
    rows = []
    for rec in records:
        if len(rec) <= max_i:
            continue
        sym = f"{str(rec[0]).strip()}{suffix}"
        if sym not in allowed_codes:
            continue
        pe = _parse_float(rec[idx_pe]) if idx_pe >= 0 else np.nan
        pb = _parse_float(rec[idx_pb]) if idx_pb >= 0 else np.nan
        yld = _parse_float(rec[idx_yld]) if idx_yld >= 0 else np.nan
        ep = float(np.clip(1.0 / pe, 0.0, 0.25)) if (np.isfinite(pe) and pe > 0) else np.nan
        roe = float(np.clip(pb / pe, -0.80, 1.20)) if (np.isfinite(pe) and pe > 0 and np.isfinite(pb) and pb > 0) else np.nan
        rows.append({
            "ticker": sym,
            "market": "TW",
            "available_date": avail,
            "period_end": snap_date,
            "rev_yoy": np.nan,
            "fcf_margin": np.nan,
            "rule40_pct": np.nan,
            "gross_margin_chg": np.nan,
            "roe": roe,
            "ttm_net_income": np.nan,
            "shares_out": np.nan,
            "ttm_eps": np.nan,
            "ep_yield": ep,
            "div_yield": float(np.clip(yld / 100.0, 0.0, 0.12)) if np.isfinite(yld) else np.nan,
        })
    return pd.DataFrame(rows)


def _fetch_twse_month_snapshot(target_dt: pd.Timestamp, allowed_codes: set, cache_dir: Path, session: requests.Session) -> pd.DataFrame:
    cfile = cache_dir / f"twse_{target_dt.strftime('%Y%m')}.json"
    data_obj, snap_date = None, None
    if cfile.exists():
        cached = json.loads(cfile.read_text(encoding="utf-8"))
        data_obj, snap_date = cached.get("data"), cached.get("date")
    else:
        for offset in (0, 1, -1, 2, -2, 3, -3, 4):
            dt = target_dt + pd.Timedelta(days=offset)
            if dt.weekday() >= 5:
                continue
            url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={dt.strftime('%Y%m%d')}&selectType=ALL&response=json"
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200:
                    j = r.json()
                    if j.get("stat") == "OK" and j.get("data"):
                        data_obj, snap_date = j, dt.strftime("%Y-%m-%d")
                        cfile.write_text(json.dumps({"date": snap_date, "data": j}), encoding="utf-8")
                        break
            except Exception:
                pass
            time.sleep(0.35)
    if not data_obj or not snap_date:
        return pd.DataFrame()
    return _parse_tw_rows(data_obj.get("data", []), data_obj.get("fields", []), ".TW", snap_date, allowed_codes)


def _fetch_tpex_month_snapshot(target_dt: pd.Timestamp, allowed_codes: set, cache_dir: Path, session: requests.Session) -> pd.DataFrame:
    cfile = cache_dir / f"tpex_{target_dt.strftime('%Y%m')}.json"
    table, snap_date = None, None
    if cfile.exists():
        cached = json.loads(cfile.read_text(encoding="utf-8"))
        table, snap_date = cached.get("table"), cached.get("date")
    else:
        for offset in (0, 1, -1, 2, -2, 3, -3, 4):
            dt = target_dt + pd.Timedelta(days=offset)
            if dt.weekday() >= 5:
                continue
            url = f"https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate?date={dt.strftime('%Y/%m/%d')}&response=json"
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200:
                    tables = r.json().get("tables") or []
                    if tables and tables[0].get("data"):
                        table, snap_date = tables[0], dt.strftime("%Y-%m-%d")
                        cfile.write_text(json.dumps({"date": snap_date, "table": table}), encoding="utf-8")
                        break
            except Exception:
                pass
            time.sleep(0.35)
    if not table or not snap_date:
        return pd.DataFrame()
    return _parse_tw_rows(table.get("data", []), table.get("fields", []), ".TWO", snap_date, allowed_codes)


def fetch_tw_exchange_pit(
    tw_tickers: Sequence[str],
    cache_dir: Path,
    start_date: str = "2019-08-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch monthly Point-in-Time TWSE + TPEx exchange valuation snapshots for all TW tickers."""
    tw_cache = cache_dir / "tw_exchange"
    tw_cache.mkdir(parents=True, exist_ok=True)
    allowed = set(tw_tickers)
    end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    months = pd.date_range(pd.Timestamp(start_date), end_ts, freq="MS") + pd.Timedelta(days=14)
    months = months[months <= end_ts]
    session = requests.Session()
    session.headers.update(TW_HEADERS)
    frames: List[pd.DataFrame] = []
    for m_dt in months:
        for fn in (_fetch_twse_month_snapshot, _fetch_tpex_month_snapshot):
            df_m = fn(m_dt, allowed, tw_cache, session)
            if not df_m.empty:
                frames.append(df_m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_split_lookup(splits_df: Optional[pd.DataFrame]) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
    lookup: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
    if splits_df is None or splits_df.empty:
        return lookup
    for _, row in splits_df.iterrows():
        sym = str(row["ticker"]).strip()
        ratio = float(row["split_ratio"])
        if sym and np.isfinite(ratio) and ratio > 0 and abs(ratio - 1.0) >= 1e-6:
            lookup.setdefault(sym, []).append((pd.Timestamp(row["split_date"]), ratio))
    for sym in lookup:
        lookup[sym].sort(key=lambda x: x[0])
    return lookup


def _cumulative_split_factor(events: Sequence[Tuple[pd.Timestamp, float]], start_ex: pd.Timestamp, end_in: pd.Timestamp) -> float:
    factor = 1.0
    for s_dt, ratio in events:
        if start_ex < s_dt <= end_in:
            factor *= ratio
    return factor


def _apply_split_scaling(
    values: pd.Series,
    tickers: pd.Series,
    anchors: pd.Series,
    panel_end: pd.Timestamp,
    split_lookup: Dict[str, List[Tuple[pd.Timestamp, float]]],
    *,
    multiply: bool,
) -> pd.Series:
    adj = values.astype(float).copy()
    for sym, evs in split_lookup.items():
        m_sym = (tickers == sym) & adj.notna() & anchors.notna()
        if m_sym.any():
            factors = np.asarray(
                [_cumulative_split_factor(evs, pd.Timestamp(a), panel_end) for a in anchors.loc[m_sym]],
                dtype=float,
            )
            adj.loc[m_sym] = adj.loc[m_sym].to_numpy() * factors if multiply else adj.loc[m_sym].to_numpy() / factors
    return adj


def _compute_split_immune_us_ep(
    merged: pd.DataFrame,
    has_period_end: bool,
    splits_df: Optional[pd.DataFrame],
    allow_adjusted_close_ep: bool,
) -> pd.Series:
    """Compute split-immune US E/P aligning SEC shares/EPS to Yahoo's split-adjusted Close basis."""
    split_lookup = _build_split_lookup(splits_df)
    has_raw = "raw_close_usd" in merged.columns and merged["raw_close_usd"].notna().any()
    px_col = "raw_close_usd" if has_raw else ("close_usd" if "close_usd" in merged.columns else None)
    if px_col is None:
        return pd.Series(np.nan, index=merged.index, dtype=float)

    # ASC 260-10-55-12: SEC filings filed AFTER a split retroactively restate share counts & EPS
    # for all periods in that filing (including prior period_end). Anchor split alignment to available_date.
    split_anchor = merged["available_date"]
    panel_end = pd.Timestamp(merged["date"].max())
    us_ep = pd.Series(np.nan, index=merged.index, dtype=float)

    if "ttm_net_income" in merged.columns and "shares_out" in merged.columns and (has_raw or split_lookup):
        shares_adj = (
            _apply_split_scaling(merged["shares_out"], merged["ticker"], split_anchor, panel_end, split_lookup, multiply=True)
            if split_lookup
            else merged["shares_out"].astype(float)
        )
        mkt_cap = (shares_adj * merged[px_col]).replace(0.0, np.nan)
        us_ep = (merged["ttm_net_income"] / mkt_cap).clip(lower=0.0, upper=0.25)

    if "ttm_eps" in merged.columns and (split_lookup or allow_adjusted_close_ep):
        eps_adj = (
            _apply_split_scaling(merged["ttm_eps"], merged["ticker"], split_anchor, panel_end, split_lookup, multiply=False)
            if split_lookup
            else merged["ttm_eps"].astype(float)
        )
        eps_ep = (eps_adj / merged[px_col].replace(0.0, np.nan)).clip(lower=0.0, upper=0.25)
        us_ep = us_ep.fillna(eps_ep)
    return us_ep


def _standardize_pit_zscores(merged: pd.DataFrame) -> pd.DataFrame:
    raw_to_z = {
        "rule40_pct": "pit_rule40_z",
        "rev_yoy": "pit_rev_yoy_z",
        "gross_margin_chg": "pit_gross_margin_chg_z",
        "roe": "pit_roe_z",
        "ep_yield": "pit_ep_z",
        "fcf_margin": "pit_fcf_margin_z",
    }
    for raw_c, z_c in raw_to_z.items():
        if raw_c not in merged.columns or merged[raw_c].isna().all():
            continue
        grp_ind = merged.groupby(["date", "market", "industry"])[raw_c]
        z_ind = (merged[raw_c] - grp_ind.transform("mean")) / grp_ind.transform("std").replace(0.0, np.nan)
        grp_mkt = merged.groupby(["date", "market"])[raw_c]
        z_mkt = (merged[raw_c] - grp_mkt.transform("mean")) / grp_mkt.transform("std").replace(0.0, np.nan)
        merged[z_c] = z_ind.where(grp_ind.transform("count") >= 4, z_mkt).clip(-3.0, 3.0).astype(np.float32)
    return merged


def merge_pit_into_panel(
    panel: pd.DataFrame,
    pit_df: pd.DataFrame,
    *,
    max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
    max_period_age_days: int = DEFAULT_MAX_PERIOD_AGE_DAYS,
    splits_df: Optional[pd.DataFrame] = None,
    allow_adjusted_close_ep: bool = False,
) -> pd.DataFrame:
    """Strictly Point-in-Time backward merge with 200-day staleness expiration and split-immune US E/P."""
    if pit_df is None or pit_df.empty:
        return panel

    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    left = left.sort_values("date").reset_index(drop=True)

    right = pit_df.copy()
    right["available_date"] = pd.to_datetime(right["available_date"]).astype("datetime64[ns]")
    has_pe, has_sd = "period_end" in right.columns, "shares_date" in right.columns
    for dt_c in [c for c in ("period_end", "shares_date") if c in right.columns]:
        right[dt_c] = pd.to_datetime(right[dt_c], errors="coerce").astype("datetime64[ns]")

    raw_fields = ("rev_yoy", "fcf_margin", "rule40_pct", "gross_margin_chg", "roe", "ttm_net_income", "shares_out", "ttm_eps", "ep_yield")
    keep_raw = [c for c in raw_fields if c in right.columns]
    extra_dt = [c for c in ("period_end", "shares_date") if c in right.columns]
    right = right[["ticker", "available_date", *extra_dt, *keep_raw]].sort_values("available_date").reset_index(drop=True)

    merged = pd.merge_asof(
        left, right, left_on="date", right_on="available_date", by="ticker",
        direction="backward", allow_exact_matches=True, tolerance=pd.Timedelta(days=max_staleness_days),
    )
    valid_ad = merged["available_date"].notna()
    if valid_ad.any() and (merged.loc[valid_ad, "available_date"] > merged.loc[valid_ad, "date"]).any():
        raise AssertionError("Look-ahead violation in PIT fundamentals merge!")
    if has_pe:
        stale_pe = merged["period_end"].notna() & ((merged["date"] - merged["period_end"]) > pd.Timedelta(days=max_period_age_days))
        if stale_pe.any():
            merged.loc[stale_pe, keep_raw] = np.nan

    us_ep = _compute_split_immune_us_ep(merged, has_pe, splits_df, allow_adjusted_close_ep)
    merged["ep_yield"] = merged["ep_yield"].fillna(us_ep) if "ep_yield" in merged.columns else us_ep
    merged = _standardize_pit_zscores(merged)
    merged = merged.drop(columns=[c for c in ["available_date", *extra_dt, *keep_raw] if c in merged.columns])
    return merged.sort_values(["date", "market", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Point-in-Time Historical Fundamentals (US SEC EDGAR + TW TWSE/TPEx)")
    parser.add_argument("--panel", type=Path, required=True, help="Existing price panel (.parquet or .csv.gz)")
    parser.add_argument("--cache-dir", type=Path, default=Path("kaggle_dataset/pit_cache"), help="Local cache for SEC/TWSE raw responses")
    parser.add_argument("--out-pit", type=Path, default=Path("kaggle_dataset/pit_fundamentals_us_tw.parquet"), help="Output raw PIT fundamentals file")
    parser.add_argument("--out-panel", type=Path, default=Path("kaggle_dataset/snowball_us_tw_2000_pit.parquet"), help="Output enriched panel file")
    parser.add_argument("--max-staleness-days", type=int, default=DEFAULT_MAX_STALENESS_DAYS, help="Max filing age in days (default: 200)")
    parser.add_argument("--offline-cache-only", action="store_true", help="Re-parse cached SEC JSONs without hitting network")
    parser.add_argument("--skip-us", action="store_true", help="Skip US SEC EDGAR fetch")
    parser.add_argument("--skip-tw", action="store_true", help="Skip TW TWSE/TPEx fetch")
    args = parser.parse_args()

    panel = load_dataframe(args.panel)
    us_tickers = sorted(panel.loc[panel["market"] == "US", "ticker"].unique().tolist())
    tw_tickers = sorted(panel.loc[panel["market"] == "TW", "ticker"].unique().tolist())
    start_str = (panel["date"].min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end_str = panel["date"].max().strftime("%Y-%m-%d")

    parts: List[pd.DataFrame] = []
    if not args.skip_us and us_tickers:
        df_us = fetch_us_sec_pit(
            us_tickers,
            cache_dir=args.cache_dir,
            min_year=max(2017, panel["date"].min().year - 1),
            offline_cache_only=args.offline_cache_only,
        )
        if not df_us.empty:
            parts.append(df_us)
    if not args.skip_tw and tw_tickers:
        df_tw = fetch_tw_exchange_pit(tw_tickers, cache_dir=args.cache_dir, start_date=start_str, end_date=end_str)
        if not df_tw.empty:
            parts.append(df_tw)

    if not parts:
        raise RuntimeError("No PIT fundamental records fetched.")

    pit_df = pd.concat(parts, ignore_index=True)
    saved_pit = save_dataframe(pit_df, args.out_pit)
    enriched = merge_pit_into_panel(panel, pit_df, max_staleness_days=args.max_staleness_days)
    saved_panel = save_dataframe(enriched, args.out_panel)
    print(f"Saved PIT: {saved_pit} ({len(pit_df):,} rows) -> Enriched panel: {saved_panel} {enriched.shape}")


if __name__ == "__main__":
    main()
