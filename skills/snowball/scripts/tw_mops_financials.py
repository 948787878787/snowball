"""MOPS quarterly financial statements (all listed/OTC companies per season) and PIT quality factors.

Statements are fetched from the MOPS summary reports (t163sb04 income, t163sb05 balance sheet).
Values are year-to-date cumulative as reported; quality factors compare the same season year over
year, so the cumulative basis cancels out.

Point-in-time: a season becomes known the day after its statutory filing deadline
(Q1 May 15, Q2 Aug 14, Q3 Nov 14, Q4 Mar 31 of the next year).

Quality factors (signs are fixed in the v8 protocol, not here):
  q_gm_yoy        gross margin minus the same season last year (non-financials only)
  q_roa_yoy       net income / total assets minus the same season last year
  q_asset_growth  total assets year-over-year growth
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger("tw_mops_financials")
MOPS_URL = "https://mopsov.twse.com.tw/mops/web/ajax_{report}"
REPORTS = {"income": "t163sb04", "balance": "t163sb05"}
MARKETS = ("sii", "otc")
PAUSE_SEC = 3.0
HTTP_TIMEOUT_SEC = 60
_DEADLINE = {1: (5, 15), 2: (8, 14), 3: (11, 14)}
_COLUMN_MAP = {
    "income": {
        "revenue": ("營業收入", "收入", "淨收益"),
        "gross_profit": ("營業毛利（毛損）淨額", "營業毛利（毛損）"),
        "net_income": ("淨利（淨損）歸屬於母公司業主", "淨利歸屬於母公司業主"),
    },
    "balance": {
        "total_assets": ("資產總計", "資產總額"),
        "equity": ("歸屬於母公司業主之權益合計", "歸屬於母公司業主權益合計"),
    },
}


def available_date(year: int, season: int) -> pd.Timestamp:
    if season == 4:
        return pd.Timestamp(year + 1, 3, 31) + pd.Timedelta(days=1)
    month, day = _DEADLINE[season]
    return pd.Timestamp(year, month, day) + pd.Timedelta(days=1)


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def parse_statement(html: str, kind: str) -> pd.DataFrame:
    """Parse every industry-format table in one MOPS summary page into (stock_id, mapped columns)."""
    wanted = _COLUMN_MAP[kind]
    frames = []
    for tbl in pd.read_html(io.StringIO(html)):
        cols = [str(c).replace(" ", "") for c in tbl.columns]
        if "公司代號" not in cols:
            continue
        tbl = tbl.set_axis(cols, axis=1)
        out = pd.DataFrame({"stock_id": tbl["公司代號"].astype(str).str.strip()})
        for name, candidates in wanted.items():
            src = next((c for c in candidates if c in tbl.columns), None)
            out[name] = _number(tbl[src]) if src else np.nan
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["stock_id", *wanted])
    full = pd.concat(frames, ignore_index=True)
    return full.loc[full["stock_id"].str.match(r"^[1-9]\d{3}$")].drop_duplicates("stock_id").reset_index(drop=True)


def _cache_path(cache_dir: Path, kind: str, market: str, year: int, season: int) -> Path:
    return cache_dir / kind / f"{market}_{year}Q{season}.html"


def fetch_page(cache_dir: Path, kind: str, market: str, year: int, season: int) -> Optional[str]:
    path = _cache_path(cache_dir, kind, market, year, season)
    if path.exists():
        return path.read_text(encoding="utf-8")
    form = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "TYPEK": market,
            "year": year - 1911, "season": f"{season:02d}"}
    try:
        resp = requests.post(MOPS_URL.format(report=REPORTS[kind]), data=form, timeout=HTTP_TIMEOUT_SEC,
                             headers={"User-Agent": "Mozilla/5.0 (research data client)"})
        resp.raise_for_status()
    except requests.RequestException as err:
        LOG.warning("MOPS fetch failed %s %s %dQ%d: %s", kind, market, year, season, err)
        return None
    finally:
        time.sleep(PAUSE_SEC)
    resp.encoding = "utf-8"
    if "公司" not in resp.text:  # throttled or not yet published: do not cache
        LOG.warning("MOPS returned no table %s %s %dQ%d", kind, market, year, season)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text, encoding="utf-8")
    return resp.text


def load_statements(cache_dir: Path, seasons: Iterable[tuple]) -> pd.DataFrame:
    """Fetch (cached) and merge income + balance for every (year, season); returns one row per stock-season."""
    rows: List[pd.DataFrame] = []
    for year, season in seasons:
        parts: Dict[str, pd.DataFrame] = {}
        for kind in REPORTS:
            pages = [fetch_page(cache_dir, kind, m, year, season) for m in MARKETS]
            frames = [parse_statement(p, kind) for p in pages if p]
            parts[kind] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["stock_id"])
        merged = parts["income"].merge(parts["balance"], on="stock_id", how="outer")
        rows.append(merged.assign(year=year, season=season))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def quality_factors(statements: pd.DataFrame) -> pd.DataFrame:
    """Same-season year-over-year quality factors with each row's PIT availability date."""
    q = statements.sort_values(["stock_id", "season", "year"]).copy()
    q["gm"] = q["gross_profit"] / q["revenue"].where(q["revenue"] > 0)
    q["roa"] = q["net_income"] / q["total_assets"].where(q["total_assets"] > 0)
    prev = q.groupby(["stock_id", "season"])[["gm", "roa", "total_assets", "year"]].shift(1)
    same_gap = prev["year"] == q["year"] - 1
    out = q[["stock_id", "year", "season"]].copy()
    out["q_gm_yoy"] = (q["gm"] - prev["gm"]).where(same_gap)
    out["q_roa_yoy"] = (q["roa"] - prev["roa"]).where(same_gap)
    out["q_asset_growth"] = (q["total_assets"] / prev["total_assets"].where(prev["total_assets"] > 0) - 1.0).where(same_gap)
    out["available"] = [available_date(int(y), int(s)) for y, s in zip(out["year"], out["season"])]
    return out.sort_values(["available", "stock_id"]).reset_index(drop=True)


def merge_asof_quality(frame: pd.DataFrame, quality: pd.DataFrame, max_staleness_days: int = 200) -> pd.DataFrame:
    """As-of merge on (stock_id, date >= available); values older than `max_staleness_days` become NaN."""
    left = frame.assign(stock_id=frame["stock_id"].astype(object),
                        date=pd.to_datetime(frame["date"]).astype("datetime64[ns]")).sort_values("date")
    right = quality.assign(stock_id=quality["stock_id"].astype(object),
                           available=pd.to_datetime(quality["available"]).astype("datetime64[ns]")).sort_values("available")
    merged = pd.merge_asof(left, right, left_on="date", right_on="available", by="stock_id", direction="backward")
    stale = (merged["date"] - merged["available"]).dt.days > max_staleness_days
    cols = ["q_gm_yoy", "q_roa_yoy", "q_asset_growth"]
    merged.loc[stale, cols] = np.nan
    return merged.drop(columns=["year", "season", "available"])


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPS quarterly statements -> PIT quality factors")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=pd.Timestamp.today().year)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = pd.Timestamp.today().normalize()
    seasons = [(y, s) for y in range(args.start_year, args.end_year + 1) for s in (1, 2, 3, 4)
               if available_date(y, s) <= today]
    quality = quality_factors(load_statements(args.cache_dir, seasons))
    quality.to_parquet(args.out, index=False)
    LOG.info("Saved %s rows=%d stocks=%d", args.out, len(quality), quality["stock_id"].nunique())


if __name__ == "__main__":
    main()
