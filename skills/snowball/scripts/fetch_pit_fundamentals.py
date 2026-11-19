"""Point-in-Time (PIT) Historical Fundamentals Fetcher for US (SEC EDGAR XBRL) & TW (TWSE/TPEx).

Builds a strictly look-ahead-free historical fundamentals dataset aligned by exact filing/publication
date (`available_date = filed_date + 1 day`) to enrich `snowball_us_tw_2000.parquet`:

1. US Stocks (~980 tickers) — SEC EDGAR Official Company Facts XBRL API (100% free, no API key):
   - Extracts exact `filed` dates from 10-Q / 10-K / 20-F filings
   - Computes PIT `rev_yoy`, `fcf_margin`, `rule40_pct` (Sequoia Rule of 40),
     `gross_margin_chg` (Microeconomic Lerner pricing power trend), `roe`, and `ttm_eps`
     (combined with panel `close_usd` at date `t` to form exact PIT `ep_yield`).

2. TW Stocks (~990 tickers) — Official TWSE (`BWIBBU_d`) + TPEx (`peQryDate`) Bulk Monthly Snapshots:
   - Fetches ALL ~1,900 TWSE + TPEx stocks in 2 JSON calls per month (100% free, no API key)
   - Extracts exact PIT `ep_yield = 1/PER`, `roe = PBR/PER` (accounting identity EPS/BVPS),
     and `div_yield` (capped at 12%).

3. Strict Backward Merge (`merge_pit_into_panel`):
   - Uses `pd.merge_asof(..., direction="backward")` enforcing `available_date <= date`
   - Cross-sectionally standardizes each metric within `(date, market, industry)` -> `pit_*_z`

Usage:
    python fetch_pit_fundamentals.py \
      --panel ./kaggle_dataset/snowball_us_tw_2000.parquet \
      --out-pit ./kaggle_dataset/pit_fundamentals_us_tw.parquet \
      --out-panel ./kaggle_dataset/snowball_us_tw_2000_pit.parquet
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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

SEC_HEADERS = {
    "User-Agent": "SnowballQuantResearch/1.0 (open-source-research@snowball-quant.org)",
    "Accept-Encoding": "gzip, deflate",
}
TW_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SnowballQuantResearch/1.0)",
    "Accept": "application/json, text/plain, */*",
}

# XBRL US-GAAP / IFRS tag candidates (ordered by coverage & specificity)
REV_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "InterestAndDividendIncomeOperating",
    "TotalRevenuesAndOtherIncome",
)
GP_TAGS = ("GrossProfit",)
OPINC_TAGS = ("OperatingIncomeLoss",)
OCF_TAGS = ("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
NI_TAGS = ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic")
EQ_TAGS = ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")

VALID_FORMS = frozenset({"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F"})


class _RateLimiter:
    """Thread-safe token interval limiter (keeps SEC requests strictly <= 7 req/sec)."""

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
    s = str(val).strip().replace(",", "").replace("--", "").replace("-", "" if str(val).strip() == "-" else "-")
    if not s or s in ("N/A", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _extract_xbrl_series(
    facts_us_gaap: dict,
    tags: Sequence[str],
    unit_key: str = "USD",
    is_duration: bool = True,
) -> pd.DataFrame:
    """Extract clean historical XBRL observations (start, end, filed, val, days) for the best matching tag."""
    best_rows: List[dict] = []
    for tag in tags:
        node = facts_us_gaap.get(tag)
        if not isinstance(node, dict):
            continue
        units = node.get("units", {})
        items = units.get(unit_key)
        if not items and unit_key == "USD":
            # Fallback to first currency unit (e.g., USD/shares or EUR)
            for k, v in units.items():
                if "USD" in k:
                    items = v
                    break
        if not items:
            continue
        rows = []
        for it in items:
            form = str(it.get("form", "")).upper()
            if form not in VALID_FORMS:
                continue
            filed = it.get("filed")
            end = it.get("end")
            val = it.get("val")
            if not filed or not end or val is None:
                continue
            if is_duration:
                start = it.get("start")
                if not start:
                    continue
                try:
                    d0 = pd.Timestamp(start)
                    d1 = pd.Timestamp(end)
                    days = (d1 - d0).days
                except Exception:
                    continue
                # Keep single-quarter (~70–115d) and full-year (~330–400d) flows
                if not ((65 <= days <= 115) or (330 <= days <= 400)):
                    continue
                rows.append({"end": d1, "filed": pd.Timestamp(filed), "val": float(val), "days": int(days)})
            else:
                try:
                    d1 = pd.Timestamp(end)
                    rows.append({"end": d1, "filed": pd.Timestamp(filed), "val": float(val), "days": 0})
                except Exception:
                    continue
        if len(rows) > len(best_rows):
            best_rows = rows

    if not best_rows:
        return pd.DataFrame(columns=["end", "filed", "val", "days"])

    df = pd.DataFrame(best_rows)
    # For each period (end, is_annual), keep the FIRST filing (point-in-time, avoids future restatement look-ahead!)
    df["is_annual"] = df["days"] >= 300
    df = df.sort_values("filed").drop_duplicates(subset=["end", "is_annual"], keep="first")
    return df.sort_values(["filed", "end"]).reset_index(drop=True)


def parse_sec_company_facts(ticker: str, payload: dict, min_year: int = 2018) -> pd.DataFrame:
    """Convert one SEC EDGAR `companyfacts` JSON into strictly Point-in-Time fundamental snapshots."""
    facts = payload.get("facts", {})
    gaap = facts.get("us-gaap") or facts.get("ifrs-full") or {}
    if not isinstance(gaap, dict) or not gaap:
        return pd.DataFrame()

    rev_df = _extract_xbrl_series(gaap, REV_TAGS, unit_key="USD", is_duration=True)
    if rev_df.empty:
        return pd.DataFrame()

    gp_df = _extract_xbrl_series(gaap, GP_TAGS, unit_key="USD", is_duration=True)
    op_df = _extract_xbrl_series(gaap, OPINC_TAGS, unit_key="USD", is_duration=True)
    ocf_df = _extract_xbrl_series(gaap, OCF_TAGS, unit_key="USD", is_duration=True)
    capex_df = _extract_xbrl_series(gaap, CAPEX_TAGS, unit_key="USD", is_duration=True)
    ni_df = _extract_xbrl_series(gaap, NI_TAGS, unit_key="USD", is_duration=True)
    eq_df = _extract_xbrl_series(gaap, EQ_TAGS, unit_key="USD", is_duration=False)
    eps_df = _extract_xbrl_series(gaap, EPS_TAGS, unit_key="USD/shares", is_duration=True)

    # Prefer quarterly rows for high-frequency PIT updates; supplement with annual rows
    def _latest_asof(src: pd.DataFrame, end_dt: pd.Timestamp, filed_dt: pd.Timestamp, annual: bool) -> float:
        if src.empty:
            return np.nan
        sub = src[(src["end"] == end_dt) & (src["filed"] <= filed_dt) & (src["is_annual"] == annual)]
        if sub.empty:
            return np.nan
        return float(sub["val"].iloc[-1])

    def _prior_year_val(src: pd.DataFrame, end_dt: pd.Timestamp, filed_dt: pd.Timestamp, annual: bool) -> float:
        if src.empty:
            return np.nan
        target_end = end_dt - pd.Timedelta(days=365)
        sub = src[
            (src["filed"] <= filed_dt)
            & (src["is_annual"] == annual)
            & ((src["end"] - target_end).abs() <= pd.Timedelta(days=25))
        ]
        if sub.empty:
            return np.nan
        return float(sub["val"].iloc[-1])

    def _latest_balance(src: pd.DataFrame, filed_dt: pd.Timestamp) -> float:
        if src.empty:
            return np.nan
        sub = src[src["filed"] <= filed_dt]
        if sub.empty:
            return np.nan
        return float(sub["val"].iloc[-1])

    def _ttm_sum(src: pd.DataFrame, filed_dt: pd.Timestamp) -> float:
        if src.empty:
            return np.nan
        sub = src[src["filed"] <= filed_dt]
        if sub.empty:
            return np.nan
        q_sub = sub[~sub["is_annual"]].sort_values("end").drop_duplicates("end", keep="last")
        if len(q_sub) >= 4:
            last4 = q_sub.iloc[-4:]
            if (last4["end"].iloc[-1] - last4["end"].iloc[0]).days <= 320:
                return float(last4["val"].sum())
        a_sub = sub[sub["is_annual"]].sort_values("end")
        if not a_sub.empty:
            return float(a_sub["val"].iloc[-1])
        return np.nan

    records: List[dict] = []
    # Keep filings from min_year - 1 onward for clean YoY lookups, emit >= min_year
    cand = rev_df[rev_df["filed"] >= pd.Timestamp(f"{min_year}-01-01")].copy()
    for _, row in cand.iterrows():
        end_dt = row["end"]
        filed_dt = row["filed"]
        is_ann = bool(row["is_annual"])
        rev_now = float(row["val"])
        if rev_now <= 0:
            continue

        rev_prev = _prior_year_val(rev_df, end_dt, filed_dt, is_ann)
        rev_yoy = (rev_now / rev_prev - 1.0) if (np.isfinite(rev_prev) and rev_prev > 0) else np.nan
        if np.isfinite(rev_yoy):
            rev_yoy = float(np.clip(rev_yoy, -0.80, 2.00))

        # Gross margin & YoY Lerner markup change
        gp_now = _latest_asof(gp_df, end_dt, filed_dt, is_ann)
        gp_prev = _prior_year_val(gp_df, end_dt, filed_dt, is_ann)
        gm_now = (gp_now / rev_now) if np.isfinite(gp_now) else np.nan
        gm_prev = (gp_prev / rev_prev) if (np.isfinite(gp_prev) and np.isfinite(rev_prev) and rev_prev > 0) else np.nan
        gm_chg = float(np.clip(gm_now - gm_prev, -0.25, 0.25)) if (np.isfinite(gm_now) and np.isfinite(gm_prev)) else np.nan

        # FCF Margin (fallback to Operating Margin) -> Sequoia Rule of 40
        ocf_now = _latest_asof(ocf_df, end_dt, filed_dt, is_ann)
        capex_now = _latest_asof(capex_df, end_dt, filed_dt, is_ann)
        op_now = _latest_asof(op_df, end_dt, filed_dt, is_ann)
        if np.isfinite(ocf_now):
            fcf_val = ocf_now - (abs(capex_now) if np.isfinite(capex_now) else 0.0)
            fcf_margin = float(np.clip(fcf_val / rev_now, -0.60, 0.60))
        elif np.isfinite(op_now):
            fcf_margin = float(np.clip(op_now / rev_now, -0.60, 0.60))
        else:
            fcf_margin = np.nan

        rule40_pct = (
            float((rev_yoy + fcf_margin) * 100.0)
            if (np.isfinite(rev_yoy) and np.isfinite(fcf_margin))
            else np.nan
        )

        # ROE (TTM Net Income / Latest Stockholders' Equity)
        ttm_ni = _ttm_sum(ni_df, filed_dt)
        eq_now = _latest_balance(eq_df, filed_dt)
        roe = float(np.clip(ttm_ni / eq_now, -0.80, 1.20)) if (np.isfinite(ttm_ni) and np.isfinite(eq_now) and eq_now > 0) else np.nan

        # TTM EPS (combined with point-in-time price during panel merge -> exact PIT E/P)
        ttm_eps = _ttm_sum(eps_df, filed_dt)

        records.append({
            "ticker": ticker,
            "market": "US",
            "available_date": (filed_dt + pd.Timedelta(days=1)).date().isoformat(),
            "rev_yoy": rev_yoy,
            "fcf_margin": fcf_margin,
            "rule40_pct": rule40_pct,
            "gross_margin_chg": gm_chg,
            "roe": roe,
            "ttm_eps": ttm_eps,
            "ep_yield": np.nan,
        })

    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records)
    out = out.sort_values("available_date").drop_duplicates(subset=["ticker", "available_date"], keep="last")
    return out


def fetch_us_sec_pit(
    us_tickers: Sequence[str],
    cache_dir: Path,
    min_year: int = 2018,
    max_workers: int = 6,
) -> pd.DataFrame:
    """Fetch Point-in-Time historical XBRL fundamentals for US tickers from SEC EDGAR."""
    sec_cache = cache_dir / "sec"
    sec_cache.mkdir(parents=True, exist_ok=True)
    cik_map_file = sec_cache / "company_tickers.json"

    if cik_map_file.exists():
        cik_raw = json.loads(cik_map_file.read_text(encoding="utf-8"))
    else:
        resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=20)
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

    targets: List[Tuple[str, str]] = []
    for sym in dict.fromkeys(us_tickers):
        k = sym.upper().strip()
        cik = ticker_to_cik.get(k) or ticker_to_cik.get(k.replace("-", ".")) or ticker_to_cik.get(k.replace(".", "-"))
        if cik:
            targets.append((sym, cik))

    print(f"[US SEC EDGAR] Mapped {len(targets)}/{len(us_tickers)} US tickers to official CIKs.")
    limiter = _RateLimiter(min_interval_sec=0.14)  # <= ~7 req/sec (SEC limit is 10 req/sec)
    session = requests.Session()
    session.headers.update(SEC_HEADERS)

    def _one(sym: str, cik: str) -> pd.DataFrame:
        cfile = sec_cache / f"CIK{cik}.json"
        if cfile.exists():
            try:
                payload = json.loads(cfile.read_text(encoding="utf-8"))
                return parse_sec_company_facts(sym, payload, min_year=min_year)
            except Exception:
                pass
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

    frames: List[pd.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, sym, cik): sym for sym, cik in targets}
        for fut in as_completed(futs):
            done += 1
            try:
                df_one = fut.result()
                if not df_one.empty:
                    frames.append(df_one)
            except Exception:
                pass
            if done % 150 == 0 or done == len(targets):
                print(f"  [US SEC] processed {done}/{len(targets)} tickers ({len(frames)} with PIT fundamentals)")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_twse_month_snapshot(target_dt: pd.Timestamp, allowed_codes: set, cache_dir: Path, session: requests.Session) -> pd.DataFrame:
    """Fetch TWSE BWIBBU_d valuation snapshot (PER, PBR, DivYield -> E/P & implied ROE) on or near target_dt."""
    ym = target_dt.strftime("%Y%m")
    cfile = cache_dir / f"twse_{ym}.json"
    rows: List[dict] = []

    data_obj = None
    snap_date = None
    if cfile.exists():
        cached = json.loads(cfile.read_text(encoding="utf-8"))
        data_obj = cached.get("data")
        snap_date = cached.get("date")
    else:
        for offset in (0, 1, -1, 2, -2, 3, -3, 4):
            dt = target_dt + pd.Timedelta(days=offset)
            if dt.weekday() >= 5:
                continue
            ds = dt.strftime("%Y%m%d")
            url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={ds}&selectType=ALL&response=json"
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200:
                    j = r.json()
                    if j.get("stat") == "OK" and j.get("data"):
                        data_obj = j
                        snap_date = dt.strftime("%Y-%m-%d")
                        cfile.write_text(json.dumps({"date": snap_date, "data": j}), encoding="utf-8")
                        break
            except Exception:
                pass
            time.sleep(0.35)

    if not data_obj or not snap_date:
        return pd.DataFrame()

    fields = data_obj.get("fields", [])
    idx_code = 0
    idx_yld = next((i for i, f in enumerate(fields) if "殖利率" in f), -1)
    idx_pe = next((i for i, f in enumerate(fields) if "本益比" in f), -1)
    idx_pb = next((i for i, f in enumerate(fields) if "淨值比" in f), -1)
    avail = (pd.Timestamp(snap_date) + pd.Timedelta(days=1)).date().isoformat()

    for rec in data_obj.get("data", []):
        if len(rec) <= max(idx_pe, idx_pb, idx_yld):
            continue
        code = str(rec[idx_code]).strip()
        sym = f"{code}.TW"
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
            "rev_yoy": np.nan,
            "fcf_margin": np.nan,
            "rule40_pct": np.nan,
            "gross_margin_chg": np.nan,
            "roe": roe,
            "ttm_eps": np.nan,
            "ep_yield": ep,
            "div_yield": float(np.clip(yld / 100.0, 0.0, 0.12)) if np.isfinite(yld) else np.nan,
        })
    return pd.DataFrame(rows)


def _fetch_tpex_month_snapshot(target_dt: pd.Timestamp, allowed_codes: set, cache_dir: Path, session: requests.Session) -> pd.DataFrame:
    """Fetch TPEx peQryDate valuation snapshot (PER, PBR, DivYield -> E/P & implied ROE) on or near target_dt."""
    ym = target_dt.strftime("%Y%m")
    cfile = cache_dir / f"tpex_{ym}.json"
    rows: List[dict] = []

    table = None
    snap_date = None
    if cfile.exists():
        cached = json.loads(cfile.read_text(encoding="utf-8"))
        table = cached.get("table")
        snap_date = cached.get("date")
    else:
        for offset in (0, 1, -1, 2, -2, 3, -3, 4):
            dt = target_dt + pd.Timedelta(days=offset)
            if dt.weekday() >= 5:
                continue
            ds = dt.strftime("%Y/%m/%d")
            url = f"https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate?date={ds}&response=json"
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200:
                    j = r.json()
                    tables = j.get("tables") or []
                    if tables and tables[0].get("data"):
                        table = tables[0]
                        snap_date = dt.strftime("%Y-%m-%d")
                        cfile.write_text(json.dumps({"date": snap_date, "table": table}), encoding="utf-8")
                        break
            except Exception:
                pass
            time.sleep(0.35)

    if not table or not snap_date:
        return pd.DataFrame()

    fields = table.get("fields", [])
    idx_code = 0
    idx_pe = next((i for i, f in enumerate(fields) if "本益比" in f), -1)
    idx_yld = next((i for i, f in enumerate(fields) if "殖利率" in f), -1)
    idx_pb = next((i for i, f in enumerate(fields) if "淨值比" in f), -1)
    avail = (pd.Timestamp(snap_date) + pd.Timedelta(days=1)).date().isoformat()

    for rec in table.get("data", []):
        if len(rec) <= max(idx_pe, idx_pb, idx_yld):
            continue
        code = str(rec[idx_code]).strip()
        sym = f"{code}.TWO"
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
            "rev_yoy": np.nan,
            "fcf_margin": np.nan,
            "rule40_pct": np.nan,
            "gross_margin_chg": np.nan,
            "roe": roe,
            "ttm_eps": np.nan,
            "ep_yield": ep,
            "div_yield": float(np.clip(yld / 100.0, 0.0, 0.12)) if np.isfinite(yld) else np.nan,
        })
    return pd.DataFrame(rows)


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

    print(f"[TW Exchange] Fetching {len(months)} monthly TWSE + TPEx valuation snapshots for {len(allowed)} TW tickers...")
    session = requests.Session()
    session.headers.update(TW_HEADERS)

    frames: List[pd.DataFrame] = []
    for i, m_dt in enumerate(months, 1):
        df_twse = _fetch_twse_month_snapshot(m_dt, allowed, tw_cache, session)
        if not df_twse.empty:
            frames.append(df_twse)
        df_tpex = _fetch_tpex_month_snapshot(m_dt, allowed, tw_cache, session)
        if not df_tpex.empty:
            frames.append(df_tpex)
        if i % 20 == 0 or i == len(months):
            print(f"  [TW Exchange] processed {i}/{len(months)} months")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def merge_pit_into_panel(panel: pd.DataFrame, pit_df: pd.DataFrame) -> pd.DataFrame:
    """Strictly Point-in-Time backward merge of PIT fundamentals into weekly price panel.

    - Enforces `available_date <= date` via `pd.merge_asof(..., direction="backward")`.
    - Computes exact PIT `ep_yield = ttm_eps / close_usd` for US rows where `ttm_eps` is present.
    - Cross-sectionally z-scores each metric within `(date, market, industry)` (fallback `(date, market)`),
      clipped to `[-3, +3]`, emitting `pit_rule40_z`, `pit_rev_yoy_z`, `pit_gross_margin_chg_z`,
      `pit_roe_z`, `pit_ep_z`, `pit_fcf_margin_z`.
    """
    if pit_df is None or pit_df.empty:
        return panel

    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    left = left.sort_values("date").reset_index(drop=True)

    right = pit_df.copy()
    right["available_date"] = pd.to_datetime(right["available_date"]).astype("datetime64[ns]")
    keep_raw = [c for c in ("rev_yoy", "fcf_margin", "rule40_pct", "gross_margin_chg", "roe", "ttm_eps", "ep_yield") if c in right.columns]
    right = right[["ticker", "available_date", *keep_raw]].sort_values("available_date").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="available_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    # Hard anti-look-ahead assertion
    valid_ad = merged["available_date"].notna()
    if valid_ad.any() and (merged.loc[valid_ad, "available_date"] > merged.loc[valid_ad, "date"]).any():
        raise AssertionError("Look-ahead violation in PIT fundamentals merge!")

    # Compute exact Point-in-Time US E/P using TTM EPS / point-in-time close_usd
    if "ttm_eps" in merged.columns and "close_usd" in merged.columns:
        us_ep = (merged["ttm_eps"] / merged["close_usd"].replace(0.0, np.nan)).clip(lower=0.0, upper=0.25)
        if "ep_yield" in merged.columns:
            merged["ep_yield"] = merged["ep_yield"].fillna(us_ep)
        else:
            merged["ep_yield"] = us_ep

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
        # Within (date, market, industry) z-score with fallback to (date, market) when peer group < 4
        grp_ind = merged.groupby(["date", "market", "industry"])[raw_c]
        cnt_ind = grp_ind.transform("count")
        mu_ind = grp_ind.transform("mean")
        sd_ind = grp_ind.transform("std").replace(0.0, np.nan)
        z_ind = (merged[raw_c] - mu_ind) / sd_ind

        grp_mkt = merged.groupby(["date", "market"])[raw_c]
        mu_mkt = grp_mkt.transform("mean")
        sd_mkt = grp_mkt.transform("std").replace(0.0, np.nan)
        z_mkt = (merged[raw_c] - mu_mkt) / sd_mkt

        merged[z_c] = z_ind.where(cnt_ind >= 4, z_mkt).clip(-3.0, 3.0).astype(np.float32)

    drop_cols = ["available_date", *keep_raw]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    return merged.sort_values(["date", "market", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Point-in-Time Historical Fundamentals (US SEC EDGAR + TW TWSE/TPEx)")
    parser.add_argument("--panel", type=Path, required=True, help="Existing price panel (.parquet or .csv.gz)")
    parser.add_argument("--cache-dir", type=Path, default=Path("kaggle_dataset/pit_cache"), help="Local cache for SEC/TWSE raw responses")
    parser.add_argument("--out-pit", type=Path, default=Path("kaggle_dataset/pit_fundamentals_us_tw.parquet"), help="Output raw PIT fundamentals file")
    parser.add_argument("--out-panel", type=Path, default=Path("kaggle_dataset/snowball_us_tw_2000_pit.parquet"), help="Output enriched panel file")
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
        df_us = fetch_us_sec_pit(us_tickers, cache_dir=args.cache_dir, min_year=max(2017, panel["date"].min().year - 1))
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

    enriched = merge_pit_into_panel(panel, pit_df)
    saved_panel = save_dataframe(enriched, args.out_panel)

    pit_cols = [c for c in enriched.columns if c.startswith("pit_")]
    print("\n=== Point-in-Time Fundamentals Enriched Successfully ===")
    print(f"  Raw PIT file      : {saved_pit} ({len(pit_df):,} filing snapshots)")
    print(f"  Enriched panel    : {saved_panel} ({enriched.shape[0]:,} rows x {enriched.shape[1]} cols)")
    for c in pit_cols:
        us_cov = enriched.loc[enriched["market"] == "US", c].notna().mean() * 100.0
        tw_cov = enriched.loc[enriched["market"] == "TW", c].notna().mean() * 100.0
        print(f"    {c:<24}: US coverage={us_cov:5.1f}% | TW coverage={tw_cov:5.1f}%")


if __name__ == "__main__":
    main()
