"""Point-in-time Taiwan equity universe for Snowball v7 T2 (survivorship & selection-bias removal).

Replaces the "today's top 1,000 applied backwards" universe with:
- Candidates = every TWSE / TPEx common stock (4-digit code) listed today (FinMind `TaiwanStockInfo`)
  plus every common stock delisted since the panel start (FinMind `TaiwanStockDelisting`).
- Monthly membership: at each month-end `m`, among stocks still trading at `m`, rank by 20-day average
  traded value (close x volume) and keep the top `N`. The selection is effective only for weeks strictly
  after `m` up to the next month-end (no same-period information).
- A stock is never a member in a week where it has no price (delisted or suspended).
- Delisting settlement: the final traded close is used as exit price. If the last trade is more than
  `STALE_GAP_DAYS` before the official delisting date, a `STALE_DELIST_HAIRCUT` is applied on top.

Data caveats (reported, not hidden):
- Yahoo Finance has no history for delisted TW stocks, so their OHLCV comes from FinMind
  `TaiwanStockPrice`, which is NOT dividend-adjusted (`TaiwanStockPriceAdj` requires a paid tier).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
LISTED_TYPES = {"twse": ".TW", "tpex": ".TWO"}
STALE_GAP_DAYS = 60
STALE_DELIST_HAIRCUT = -0.30
DEFAULT_TOP_N = 1000
DEFAULT_LOOKBACK_DAYS = 20
DEFAULT_MIN_OBS = 10
_RETRIES = 3
_COMMON_CODE = r"[1-9]\d{3}"  # 4-digit codes not starting with 0 (0xxx are ETFs)

# FinMind industry_category (zh-TW) -> Snowball 13 standardized industries.
CATEGORY_TO_GROUP: Dict[str, str] = {
    "半導體業": "semiconductors_hardware", "電子零組件業": "semiconductors_hardware",
    "光電業": "semiconductors_hardware", "電腦及週邊設備業": "semiconductors_hardware",
    "通信網路業": "semiconductors_hardware", "其他電子業": "semiconductors_hardware",
    "其他電子類": "semiconductors_hardware", "電子通路業": "semiconductors_hardware",
    "電子通路類": "semiconductors_hardware", "電子工業": "semiconductors_hardware",
    "資訊服務業": "software_internet", "數位雲端": "software_internet", "數位雲端類": "software_internet",
    "電子商務業": "software_internet", "文化創意業": "software_internet",
    "金融保險": "insurance_diversified_fin", "金融業": "insurance_diversified_fin",
    "生技醫療業": "healthcare_pharma", "化學生技醫療": "healthcare_pharma",
    "食品工業": "consumer_staples", "農業科技": "consumer_staples", "農業科技業": "consumer_staples",
    "貿易百貨": "consumer_discretionary", "觀光餐旅": "consumer_discretionary",
    "觀光事業": "consumer_discretionary", "紡織纖維": "consumer_discretionary",
    "居家生活": "consumer_discretionary", "居家生活類": "consumer_discretionary",
    "運動休閒": "consumer_discretionary", "運動休閒類": "consumer_discretionary",
    "汽車工業": "automobiles_mobility", "橡膠工業": "automobiles_mobility",
    "電機機械": "industrials_machinery", "電器電纜": "industrials_machinery",
    "建材營造": "industrials_machinery", "綠能環保": "industrials_machinery",
    "綠能環保類": "industrials_machinery",
    "航運業": "transportation_logistics",
    "油電燃氣業": "energy_oil_gas",
    "化學工業": "materials_chemicals_metals", "鋼鐵工業": "materials_chemicals_metals",
    "塑膠工業": "materials_chemicals_metals", "水泥工業": "materials_chemicals_metals",
    "造紙工業": "materials_chemicals_metals", "玻璃陶瓷": "materials_chemicals_metals",
}
_GENERIC_CATEGORIES = {"電子工業", "其他", "創新板股票", "創新版股票"}


# --- FinMind I/O ---------------------------------------------------------------------------------


def fetch_finmind(dataset: str, **params: str) -> pd.DataFrame:
    """Fetch one FinMind v4 dataset; uses `FINMIND_TOKEN` from the environment when present."""
    query = {"dataset": dataset, **params}
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        query["token"] = token
    last_err: Optional[Exception] = None
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.get(FINMIND_URL, params=query, timeout=60)
            payload = resp.json()
            if resp.status_code == 200 and payload.get("status") == 200:
                return pd.DataFrame(payload.get("data", []))
            last_err = RuntimeError(f"{resp.status_code} {payload.get('msg')}")
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
        LOG.warning("FinMind %s %s attempt %d failed: %s", dataset, params, attempt, last_err)
        time.sleep(2.0 * attempt)
    raise RuntimeError(f"FinMind fetch failed: {dataset} {params}: {last_err}")


def select_common_stocks(info: pd.DataFrame) -> pd.DataFrame:
    """Return one row per listed TWSE/TPEx 4-digit common stock with Yahoo ticker and Snowball group."""
    df = info.loc[info["stock_id"].astype(str).str.fullmatch(_COMMON_CODE) & info["type"].isin(LISTED_TYPES)].copy()
    df["_generic"] = df["industry_category"].isin(_GENERIC_CATEGORIES)
    df = df.sort_values(["stock_id", "_generic"]).drop_duplicates("stock_id", keep="first")
    df["ticker"] = df["stock_id"] + df["type"].map(LISTED_TYPES)
    df["group"] = df["industry_category"].map(CATEGORY_TO_GROUP).fillna("other")
    return df[["stock_id", "ticker", "type", "group"]].sort_values("stock_id").reset_index(drop=True)


def filter_delistings(delist: pd.DataFrame, start: str) -> pd.DataFrame:
    """Keep 4-digit common stocks delisted on/after `start`; returns `stock_id`, `delist_date`."""
    df = delist.loc[delist["stock_id"].astype(str).str.fullmatch(_COMMON_CODE)].copy()
    df["delist_date"] = pd.to_datetime(df["date"])
    df = df.loc[df["delist_date"] >= pd.Timestamp(start)]
    return df[["stock_id", "delist_date"]].drop_duplicates("stock_id").sort_values("stock_id").reset_index(drop=True)


def fetch_delisted_ohlcv(
    delisted: pd.DataFrame, ticker_map: Mapping[str, str], start: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, bool]]:
    """Download unadjusted FinMind OHLCV for delisted stocks.

    Returns (close, open, volume, stale_flags) where stale_flags[ticker] is True when the last trade
    happened more than `STALE_GAP_DAYS` before the official delisting date.
    """
    closes, opens, vols, stale = {}, {}, {}, {}
    for row in delisted.itertuples(index=False):
        tkr = ticker_map.get(row.stock_id, f"{row.stock_id}.TW")
        px = fetch_finmind("TaiwanStockPrice", data_id=row.stock_id, start_date=start)
        if px.empty:
            LOG.warning("No FinMind prices for delisted %s", row.stock_id)
            continue
        px["date"] = pd.to_datetime(px["date"])
        px = px.loc[px["close"] > 0].set_index("date").sort_index()
        if px.empty:
            continue
        closes[tkr] = px["close"].astype(float)
        opens[tkr] = px["open"].astype(float)
        vols[tkr] = px["Trading_Volume"].astype(float)
        stale[tkr] = bool((row.delist_date - px.index.max()).days > STALE_GAP_DAYS)
    return pd.DataFrame(closes), pd.DataFrame(opens), pd.DataFrame(vols), stale


# --- membership mask -----------------------------------------------------------------------------


def _month_end_trading_days(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period("M")).max())


def _select_top_n_at(
    close_d: pd.DataFrame, vol_d: pd.DataFrame, at: pd.Timestamp, top_n: int, lookback_days: int, min_obs: int,
) -> set:
    window_close = close_d.loc[:at].tail(lookback_days)
    window_vol = vol_d.reindex_like(close_d).loc[:at].tail(lookback_days)
    traded_value = (window_close * window_vol).where(window_vol > 0)
    obs = traded_value.notna().sum()
    trading_now = close_d.loc[:at].tail(5).notna().any()
    eligible = (obs >= min_obs) & trading_now
    avg_value = traded_value.mean().where(eligible).dropna()
    return set(avg_value.nlargest(top_n).index)


def build_monthly_membership_mask(
    close_d: pd.DataFrame, vol_d: pd.DataFrame, weeks: Iterable[pd.Timestamp], *,
    top_n: int = DEFAULT_TOP_N, lookback_days: int = DEFAULT_LOOKBACK_DAYS, min_obs: int = DEFAULT_MIN_OBS,
) -> pd.DataFrame:
    """Boolean `week x ticker` mask: month-end top-N by traded value, effective strictly after month-end."""
    close_d = close_d.sort_index()
    weeks_idx = pd.DatetimeIndex(sorted(weeks))
    month_ends = _month_end_trading_days(close_d.index)
    selections = {m: _select_top_n_at(close_d, vol_d, m, top_n, lookback_days, min_obs) for m in month_ends}

    me_idx = pd.DatetimeIndex(month_ends)
    rows = []
    for w in weeks_idx:
        pos = int(me_idx.searchsorted(w, side="left")) - 1  # latest month-end strictly before w
        members = selections[me_idx[pos]] if pos >= 0 else set()
        rows.append([t in members for t in close_d.columns])
    mask = pd.DataFrame(rows, index=weeks_idx, columns=close_d.columns, dtype=bool)

    weekly_has_price = close_d.resample("W-FRI").count().reindex(weeks_idx).fillna(0) > 0
    return mask & weekly_has_price.reindex(columns=close_d.columns, fill_value=False)


def summarize_mask(mask: pd.DataFrame, delisted_tickers: Iterable[str]) -> pd.DataFrame:
    """Yearly constituent counts, entries/exits and member delisting events for the T2 report."""
    m = mask.astype(int)
    entries = (m.diff().fillna(0) > 0).sum(axis=1)
    exits = (m.diff().fillna(0) < 0).sum(axis=1)
    dl = [t for t in delisted_tickers if t in mask.columns]
    ever_member_delisted = mask[dl].any() if dl else pd.Series(dtype=bool)
    last_member = mask[dl].apply(lambda c: c[c].index.max() if c.any() else pd.NaT) if dl else pd.Series(dtype="datetime64[ns]")
    years = mask.index.year
    out = pd.DataFrame({
        "avg_members": mask.sum(axis=1).groupby(years).mean(),
        "entries": entries.groupby(years).sum(),
        "exits": exits.groupby(years).sum(),
    })
    out["member_delistings"] = [
        int(((last_member.dt.year == y) & ever_member_delisted).sum()) if dl else 0 for y in out.index
    ]
    return out


# --- delisting settlement ------------------------------------------------------------------------


def apply_delisting_settlement(
    next_open_w: pd.DataFrame, close_w: pd.DataFrame, delisted_stale: Mapping[str, bool],
    last_trade: Optional[Mapping[str, pd.Timestamp]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Settle delisted names at their final close so forward returns are realised, not dropped.

    The final week is the week containing the last trade (`last_trade`, required when prices were
    forward-filled, e.g. by FX conversion) or else the last week with a close. From that week on,
    next-open is held flat at the final weekly close (with `STALE_DELIST_HAIRCUT` if stale), so the
    prior week's return is realised and the final week's is 0. `final_flag` marks that final week.
    """
    filled = next_open_w.copy()
    final = pd.DataFrame(False, index=next_open_w.index, columns=next_open_w.columns)
    weeks = filled.index
    for tkr, is_stale in delisted_stale.items():
        if tkr not in filled.columns or tkr not in close_w.columns:
            continue
        final_week = _final_week(close_w[tkr], weeks, (last_trade or {}).get(tkr))
        if final_week is None:
            continue
        settle = float(close_w.at[final_week, tkr]) * (1.0 + STALE_DELIST_HAIRCUT if is_stale else 1.0)
        filled.iloc[weeks.get_loc(final_week):, filled.columns.get_loc(tkr)] = settle
        final.at[final_week, tkr] = True
    return filled, final


def _final_week(close_col: pd.Series, weeks: pd.DatetimeIndex, last_trade: Optional[pd.Timestamp]):
    if last_trade is None or pd.isna(last_trade):
        return close_col.last_valid_index()
    pos = int(weeks.searchsorted(pd.Timestamp(last_trade), side="left"))
    return weeks[pos] if pos < len(weeks) else None
