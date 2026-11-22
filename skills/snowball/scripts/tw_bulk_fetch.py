"""Bulk TWSE / TPEx chip data (all stocks per date) with a resumable raw-JSON disk cache.

Datasets (official exchange endpoints, no token needed):
  institutional : daily foreign (ex foreign dealers) and investment-trust net buy, shares
  margin        : margin-purchase balance, lots (1 lot = 1,000 shares)
  foreign_hold  : foreign + mainland holding ratio, percent of issued shares

Values are as published on date `d`; they are known after that day's close (point-in-time = d).

Usage (keep --cache-dir OUTSIDE the git repo):
  python tw_bulk_fetch.py --cache-dir ../../../../kaggle_dataset/tw_bulk_cache --start 2018-06-01 --end 2026-11-20
  python tw_bulk_fetch.py --cache-dir ... --build-out ../../../../kaggle_dataset/tw_chip_daily.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import quote

import pandas as pd
import requests

LOG = logging.getLogger("tw_bulk_fetch")
_COMMON_CODE = r"^[1-9]\d{3}$"
HOST_PAUSE_SEC = 2.5
HTTP_TIMEOUT_SEC = 30
MAX_RETRIES = 3
_HEADERS = {"User-Agent": "Mozilla/5.0 (research data client)"}

_TWSE = "https://www.twse.com.tw/rwd/zh"
_TPEX = "https://www.tpex.org.tw/www/zh-tw"
_URLS: Dict[str, Dict[str, str]] = {
    "institutional": {"twse": _TWSE + "/fund/T86?date={ymd}&selectType=ALLBUT0999&response=json",
                      "tpex": _TPEX + "/insti/dailyTrade?type=Daily&sect=EW&date={slash}&response=json"},
    "margin": {"twse": _TWSE + "/marginTrading/MI_MARGN?date={ymd}&selectType=STOCK&response=json",
               "tpex": _TPEX + "/margin/balance?date={slash}&response=json"},
    "foreign_hold": {"twse": _TWSE + "/fund/MI_QFIIS?date={ymd}&selectType=ALLBUT0999&response=json",
                     "tpex": _TPEX + "/insti/qfii?date={slash}&response=json"},
}
# (stock_id column, {output column: source column index}) per dataset and exchange
_LAYOUT = {
    ("institutional", "twse"): (0, {"foreign_net": 4, "trust_net": 10}),
    ("institutional", "tpex"): (0, {"foreign_net": 4, "trust_net": 13}),
    ("margin", "twse"): (0, {"margin_lots": 6}),
    ("margin", "tpex"): (0, {"margin_lots": 6}),
    ("foreign_hold", "twse"): (0, {"foreign_hold_pct": 7}),
    ("foreign_hold", "tpex"): (1, {"foreign_hold_pct": 7}),
}
DAILY_DATASETS = ("institutional",)
WEEKLY_DATASETS = ("margin", "foreign_hold")


def build_url(dataset: str, exchange: str, day: pd.Timestamp) -> str:
    return _URLS[dataset][exchange].format(ymd=day.strftime("%Y%m%d"), slash=quote(day.strftime("%Y/%m/%d"), safe=""))


def _to_number(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _stock_rows(payload: dict, id_col: int) -> List[list]:
    """Pick the per-stock table: the first table containing any common-stock code in its id column."""
    tables = payload.get("tables") or [payload]
    for tbl in tables:
        rows = tbl.get("data") or []
        if any(re.match(_COMMON_CODE, str(r[id_col]).strip()) for r in rows if len(r) > id_col):
            return rows
    return []


def _parse(payload: dict, dataset: str, exchange: str, day: pd.Timestamp) -> pd.DataFrame:
    id_col, cols = _LAYOUT[(dataset, exchange)]
    empty = pd.DataFrame(columns=["stock_id", "date", *cols])
    if str(payload.get("stat", "")).lower() != "ok":
        return empty
    rows = [r for r in _stock_rows(payload, id_col) if len(r) > max(cols.values())]
    if not rows:
        return empty
    frame = pd.DataFrame({"stock_id": [str(r[id_col]).strip() for r in rows],
                          **{name: [_to_number(r[i]) for r in rows] for name, i in cols.items()}})
    frame = frame.loc[frame["stock_id"].str.match(_COMMON_CODE)].reset_index(drop=True)
    return frame.assign(date=pd.Timestamp(day))[["stock_id", "date", *cols]]


def parse_institutional(payload: dict, exchange: str, day: pd.Timestamp) -> pd.DataFrame:
    return _parse(payload, "institutional", exchange, day)


def parse_margin(payload: dict, exchange: str, day: pd.Timestamp) -> pd.DataFrame:
    return _parse(payload, "margin", exchange, day)


def parse_foreign_holding(payload: dict, exchange: str, day: pd.Timestamp) -> pd.DataFrame:
    return _parse(payload, "foreign_hold", exchange, day)


PARSERS: Dict[str, Callable[[dict, str, pd.Timestamp], pd.DataFrame]] = {
    "institutional": parse_institutional, "margin": parse_margin, "foreign_hold": parse_foreign_holding,
}


def weekly_fridays(start: str, end: str) -> List[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="W-FRI"))


def _cache_path(cache_dir: Path, dataset: str, exchange: str, day: pd.Timestamp) -> Path:
    return cache_dir / dataset / exchange / f"{day.strftime('%Y%m%d')}.json"


def _get_json(url: str) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=HTTP_TIMEOUT_SEC)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as err:
            LOG.warning("Fetch failed (%d/%d) %s: %s", attempt + 1, MAX_RETRIES, url, err)
            time.sleep(HOST_PAUSE_SEC * (attempt + 2) * 4)
    return None


def fetch_one(cache_dir: Path, dataset: str, exchange: str, day: pd.Timestamp) -> Optional[dict]:
    """Return the cached payload, fetching it once if absent. Failed fetches are not cached."""
    path = _cache_path(cache_dir, dataset, exchange, day)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = _get_json(build_url(dataset, exchange, day))
    time.sleep(HOST_PAUSE_SEC)
    if payload is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _jobs(start: str, end: str) -> List[tuple]:
    days = [d for d in pd.bdate_range(start, end)]
    fridays = weekly_fridays(start, end)
    jobs = [(ds, d) for d in days for ds in DAILY_DATASETS]
    # Weekly stock variables: Friday plus the 4 prior weekdays as holiday fallbacks are fetched lazily below.
    jobs += [(ds, d) for d in fridays for ds in WEEKLY_DATASETS]
    return jobs


def _run_exchange(cache_dir: Path, exchange: str, start: str, end: str) -> int:
    done = 0
    for dataset, day in _jobs(start, end):
        payload = fetch_one(cache_dir, dataset, exchange, day)
        if dataset in WEEKLY_DATASETS and payload is not None and _parse(payload, dataset, exchange, day).empty:
            for back in range(1, 5):  # Friday holiday: take the last trading day of that week
                alt = day - pd.Timedelta(days=back)
                alt_payload = fetch_one(cache_dir, dataset, exchange, alt)
                if alt_payload is not None and not _parse(alt_payload, dataset, exchange, alt).empty:
                    break
        done += 1
        if done % 200 == 0:
            LOG.info("%s: %d jobs processed (at %s %s)", exchange, done, dataset, day.date())
    return done


def fetch_range(cache_dir: Path, start: str, end: str) -> None:
    """Fetch both exchanges in parallel (one thread per host, each host rate-limited)."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_exchange, cache_dir, ex, start, end) for ex in ("twse", "tpex")]
        for fut in futures:
            LOG.info("Finished %d jobs", fut.result())


def build_tidy(cache_dir: Path, datasets: Iterable[str] = ("institutional", "margin", "foreign_hold")) -> Dict[str, pd.DataFrame]:
    """Parse every cached payload into one tidy frame per dataset (stock_id, date, values)."""
    out: Dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        frames = []
        for exchange in ("twse", "tpex"):
            for path in sorted((cache_dir / dataset / exchange).glob("*.json")):
                day = pd.Timestamp(path.stem)
                frames.append(PARSERS[dataset](json.loads(path.read_text(encoding="utf-8")), exchange, day))
        frames = [f for f in frames if not f.empty]
        tidy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out[dataset] = tidy.drop_duplicates(["stock_id", "date"], keep="last").sort_values(["date", "stock_id"]) if not tidy.empty else tidy
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="TWSE/TPEx bulk chip data fetcher")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--start", default="2018-06-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--build-out", type=Path, default=None, help="Write tidy parquet files with this stem")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.build_out is None:
        fetch_range(args.cache_dir, args.start, args.end)
        return
    for name, frame in build_tidy(args.cache_dir).items():
        path = args.build_out.with_name(f"{args.build_out.stem}_{name}.parquet")
        frame.to_parquet(path, index=False)
        LOG.info("Saved %s rows=%d", path, len(frame))


if __name__ == "__main__":
    main()
