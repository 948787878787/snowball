"""Taiwan Monthly Revenue PIT Fetcher & Stage 4 Low-Risk Features (Snowball v7).

Implements Stage 4 features strictly obeying Point-in-Time rules:
1. Taiwan Monthly Revenue (MOPS TWSE/TPEx static monthly tables):
   - Statutory deadline is the 10th of the next month -> `available_date = next month 11th`.
   - Trailing 3-month cumulative YoY (`tw_rev_3m_yoy`, clipped [-0.80, +2.00]).
   - 3-month YoY acceleration (`tw_rev_yoy_accel`, clipped [-1.00, +1.00]).
   - 12-month revenue high (`tw_rev_12m_high`, 1.0 if current month >= 12-month max else 0.0).
2. 52-Week High Proximity (George & Hwang 2004): `close_usd / max(close_usd, 52w)` Z-score (`f_52w_high`).
3. 4w + 12w Horizon Rank Blending: `0.5 * rank(4w) + 0.5 * rank(12w)`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests

STAGE4_FEATURE_COLS: Tuple[str, ...] = (
    "pit_tw_rev_3m_yoy_z",
    "pit_tw_rev_accel_z",
    "pit_tw_rev_12m_high_z",
    "f_52w_high",
)

MOPS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SnowballQuantResearch/1.0)",
    "Accept": "text/html,application/xhtml+xml,*/*",
}


class _TableParser(HTMLParser):
    """Lightweight stdlib HTML table parser (zero external lxml/bs4 dependencies)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._in_tr = False
        self._in_cell = False
        self._cur_row: List[str] = []
        self._cur_cell: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        t = tag.lower()
        if t == "tr":
            self._in_tr = True
            self._cur_row = []
        elif self._in_tr and t in ("td", "th"):
            self._in_cell = True
            self._cur_cell = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            self._cur_row.append("".join(self._cur_cell).strip())
            self._in_cell = False
        elif t == "tr" and self._in_tr:
            if self._cur_row:
                self.rows.append(self._cur_row)
            self._in_tr = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cur_cell.append(data)


def _parse_mops_html(html_text: str, suffix: str, year: int, month: int, allowed: Optional[Set[str]]) -> List[dict]:
    parser = _TableParser()
    parser.feed(html_text)
    records: List[dict] = []
    for row in parser.rows:
        if len(row) < 3:
            continue
        code = row[0].strip()
        if not (4 <= len(code) <= 6 and code.isalnum() and code[0].isdigit()):
            continue
        sym = f"{code}{suffix}"
        if allowed is not None and sym not in allowed:
            continue
        rev_s = row[2].replace(",", "").strip()
        try:
            rev_val = float(rev_s)
        except ValueError:
            continue
        if np.isfinite(rev_val) and rev_val > 0:
            records.append({"ticker": sym, "revenue_year": year, "revenue_month": month, "revenue": rev_val})
    return records


def _fetch_one_mops_month(
    year: int, month: int, mkt_code: str, sub_type: int, cache_dir: Path, allowed: Optional[Set[str]], session: requests.Session,
) -> List[dict]:
    roc_y = year - 1911
    suffix = ".TW" if mkt_code == "sii" else ".TWO"
    cfile = cache_dir / f"{mkt_code}_{roc_y}_{month}_{sub_type}.html"
    html_text = ""
    if cfile.exists():
        html_text = cfile.read_text(encoding="utf-8", errors="ignore")
    else:
        url = f"https://mopsov.twse.com.tw/nas/t21/{mkt_code}/t21sc03_{roc_y}_{month}_{sub_type}.html"
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 500:
                html_text = r.content.decode("cp950", errors="ignore")
                cfile.write_text(html_text, encoding="utf-8")
        except Exception:
            return []
    return _parse_mops_html(html_text, suffix, year, month, allowed) if html_text else []


def fetch_mops_monthly_revenue(
    start_year: int,
    end_year: int,
    cache_dir: Path,
    *,
    tickers_filter: Optional[Sequence[str]] = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch official MOPS TWSE/TPEx monthly revenues for all Taiwan stocks with local HTML cache."""
    rev_cache = cache_dir / "tw_mops_rev"
    rev_cache.mkdir(parents=True, exist_ok=True)
    allowed = set(tickers_filter) if tickers_filter else None
    tasks = [
        (y, m, mkt, sub)
        for y in range(start_year, end_year + 1)
        for m in range(1, 13)
        for mkt in ("sii", "otc")
        for sub in (0, 1)
    ]
    session = requests.Session()
    session.headers.update(MOPS_HEADERS)
    all_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_one_mops_month, y, m, mkt, sub, rev_cache, allowed, session) for y, m, mkt, sub in tasks]
        for fut in as_completed(futs):
            try:
                all_rows.extend(fut.result())
            except Exception:
                pass
    if not all_rows:
        return pd.DataFrame(columns=["ticker", "revenue_year", "revenue_month", "revenue"])
    df = pd.DataFrame(all_rows)
    return df.drop_duplicates(subset=["ticker", "revenue_year", "revenue_month"], keep="last").reset_index(drop=True)


def _next_month_11th(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year + 1, month=1, day=11) if month == 12 else pd.Timestamp(year=year, month=month + 1, day=11)


def compute_tw_monthly_rev_features(rev_df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly Point-in-Time 3m cumulative YoY, YoY acceleration, and 12m revenue high."""
    if rev_df is None or rev_df.empty:
        return pd.DataFrame(columns=["ticker", "revenue_year", "revenue_month", "available_date", "tw_rev_3m_yoy", "tw_rev_yoy_accel", "tw_rev_12m_high"])

    df = rev_df.copy()
    df["revenue_year"] = df["revenue_year"].astype(int)
    df["revenue_month"] = df["revenue_month"].astype(int)
    df["ym_idx"] = df["revenue_year"] * 12 + df["revenue_month"]
    df["available_date"] = [_next_month_11th(y, m) for y, m in zip(df["revenue_year"], df["revenue_month"])]
    df["available_date"] = pd.to_datetime(df["available_date"]).astype("datetime64[ns]")
    df = df.sort_values(["ticker", "ym_idx"]).drop_duplicates(["ticker", "ym_idx"], keep="last")

    out_frames: List[pd.DataFrame] = []
    for _, grp in df.groupby("ticker", sort=False):
        g = grp.copy()
        rev_map = dict(zip(g["ym_idx"], g["revenue"]))
        ym = g["ym_idx"].to_numpy()
        rev = g["revenue"].to_numpy(dtype=float)

        yoy_3m = np.full(len(g), np.nan, dtype=float)
        high_12m = np.full(len(g), np.nan, dtype=float)
        for i, m_idx in enumerate(ym):
            cur3 = [rev_map.get(m_idx - k, np.nan) for k in (0, 1, 2)]
            prev3 = [rev_map.get(m_idx - 12 - k, np.nan) for k in (0, 1, 2)]
            if all(np.isfinite(cur3)) and all(np.isfinite(prev3)) and sum(prev3) > 0:
                yoy_3m[i] = float(np.clip(sum(cur3) / sum(prev3) - 1.0, -0.80, 2.00))
            past12 = [rev_map.get(m_idx - k, np.nan) for k in range(12)]
            if sum(np.isfinite(past12)) >= 12:
                high_12m[i] = 1.0 if rev[i] >= max(past12) else 0.0

        yoy_map = dict(zip(ym, yoy_3m))
        accel = np.array(
            [
                float(np.clip(yoy_3m[i] - yoy_map.get(m_idx - 3, np.nan), -1.00, 1.00))
                if (np.isfinite(yoy_3m[i]) and np.isfinite(yoy_map.get(m_idx - 3, np.nan)))
                else np.nan
                for i, m_idx in enumerate(ym)
            ],
            dtype=float,
        )
        g["tw_rev_3m_yoy"] = yoy_3m
        g["tw_rev_yoy_accel"] = accel
        g["tw_rev_12m_high"] = high_12m
        out_frames.append(g)

    res = pd.concat(out_frames, ignore_index=True)
    return res.drop(columns=["ym_idx"]).sort_values(["ticker", "available_date"]).reset_index(drop=True)


def _standardize_cols(df: pd.DataFrame, raw_to_z: Dict[str, str]) -> pd.DataFrame:
    for raw_c, z_c in raw_to_z.items():
        if raw_c not in df.columns or df[raw_c].isna().all():
            continue
        grp_ind = df.groupby(["date", "market", "industry"])[raw_c]
        z_ind = (df[raw_c] - grp_ind.transform("mean")) / grp_ind.transform("std").replace(0.0, np.nan)
        grp_mkt = df.groupby(["date", "market"])[raw_c]
        z_mkt = (df[raw_c] - grp_mkt.transform("mean")) / grp_mkt.transform("std").replace(0.0, np.nan)
        df[z_c] = z_ind.where(grp_ind.transform("count") >= 4, z_mkt).clip(-3.0, 3.0).astype(np.float32)
    return df


def merge_tw_monthly_rev_into_panel(
    panel: pd.DataFrame,
    rev_feat_df: pd.DataFrame,
    *,
    max_staleness_days: int = 65,
) -> pd.DataFrame:
    """Strictly Point-in-Time backward merge of Taiwan monthly revenue features into weekly panel."""
    if rev_feat_df is None or rev_feat_df.empty:
        return panel
    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    left = left.sort_values("date").reset_index(drop=True)

    right = rev_feat_df[["ticker", "available_date", "tw_rev_3m_yoy", "tw_rev_yoy_accel", "tw_rev_12m_high"]].copy()
    right["available_date"] = pd.to_datetime(right["available_date"]).astype("datetime64[ns]")
    right = right.sort_values("available_date").reset_index(drop=True)

    merged = pd.merge_asof(
        left, right, left_on="date", right_on="available_date", by="ticker",
        direction="backward", allow_exact_matches=True, tolerance=pd.Timedelta(days=max_staleness_days),
    )
    valid = merged["available_date"].notna()
    if valid.any() and (merged.loc[valid, "available_date"] > merged.loc[valid, "date"]).any():
        raise AssertionError("Look-ahead violation in Taiwan monthly revenue PIT merge!")

    merged = _standardize_cols(merged, {
        "tw_rev_3m_yoy": "pit_tw_rev_3m_yoy_z",
        "tw_rev_yoy_accel": "pit_tw_rev_accel_z",
        "tw_rev_12m_high": "pit_tw_rev_12m_high_z",
    })
    drop_c = ["available_date", "tw_rev_3m_yoy", "tw_rev_yoy_accel", "tw_rev_12m_high"]
    return merged.drop(columns=drop_c).sort_values(["date", "market", "ticker"]).reset_index(drop=True)


def add_52w_high_proximity(panel: pd.DataFrame, price_col: str = "close_usd") -> pd.DataFrame:
    """Add George & Hwang (2004) 52-week high proximity Z-score (`f_52w_high`), strictly causal."""
    if price_col not in panel.columns:
        return panel
    out = panel.sort_values(["ticker", "date"]).copy()
    roll_max = out.groupby("ticker")[price_col].transform(lambda s: s.rolling(52, min_periods=26).max())
    out["_raw_52w_high"] = (out[price_col] / roll_max.replace(0.0, np.nan)).clip(0.0, 1.0)
    out = _standardize_cols(out, {"_raw_52w_high": "f_52w_high"})
    return out.drop(columns=["_raw_52w_high"]).sort_values(["date", "market", "ticker"]).reset_index(drop=True)


def blend_horizon_predictions(
    oos_4w: pd.DataFrame,
    oos_12w: pd.DataFrame,
    score_col: str,
    *,
    w_4w: float = 0.5,
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """Blend 4-week and 12-week OOS model scores via cross-sectional percentile rank average."""
    target_col = out_col or f"{score_col}_blend_4w_12w"
    a = oos_4w.copy()
    b = oos_12w[["date", "ticker", score_col]].rename(columns={score_col: "_score_12w"}).copy()
    a["date"] = pd.to_datetime(a["date"]).astype("datetime64[ns]")
    b["date"] = pd.to_datetime(b["date"]).astype("datetime64[ns]")
    merged = a.merge(b, on=["date", "ticker"], how="left")
    grp_keys = ["date", "market"] if "market" in merged.columns else ["date"]
    r4 = merged.groupby(grp_keys)[score_col].rank(pct=True)
    r12 = merged.groupby(grp_keys)["_score_12w"].rank(pct=True)
    merged[target_col] = (float(w_4w) * r4 + (1.0 - float(w_4w)) * r12).fillna(r4)
    return merged.drop(columns=["_score_12w"])
