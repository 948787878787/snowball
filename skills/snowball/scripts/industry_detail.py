"""Detailed TWSE/TPEx industry classification for Snowball v8 (about 30 official categories).

The 13-group `industry` column merges every electronics category into one bucket (over half of the
TW universe), which makes within-industry demeaning and industry caps nearly ineffective. v8 uses
the exchanges' own categories instead, with TPEx spellings folded into the TWSE names.

Limitation: FinMind `TaiwanStockInfo` only carries the current category, so the latest classification
is applied to history (reclassifications are rare). Tickers without a specific category fall back to
`grp:<13-group name>`.
"""

from __future__ import annotations

import pandas as pd

from tw_pit_universe import LISTED_TYPES

_COMMON_CODE = r"[1-9]\d{3}"
_GENERIC = {"電子工業", "其他", "創新板股票", "創新版股票"}
_ALIASES = {
    "其他電子類": "其他電子業", "電子通路類": "電子通路業", "數位雲端類": "數位雲端",
    "綠能環保類": "綠能環保", "居家生活類": "居家生活", "運動休閒類": "運動休閒",
    "觀光餐旅": "觀光事業", "金融業": "金融保險", "化學生技醫療": "生技醫療業", "農業科技": "農業科技業",
}


def detailed_industry_map(info: pd.DataFrame) -> pd.DataFrame:
    """One row per listed 4-digit common stock: `ticker`, `industry_twse` (NaN when only generic)."""
    df = info.loc[info["stock_id"].astype(str).str.fullmatch(_COMMON_CODE)
                  & info["type"].isin(LISTED_TYPES)].copy()
    df["_generic"] = df["industry_category"].isin(_GENERIC)
    df = df.sort_values(["stock_id", "_generic"]).drop_duplicates("stock_id", keep="first")
    cat = df["industry_category"].replace(_ALIASES)
    df["industry_twse"] = cat.where(~df["_generic"])
    df["ticker"] = df["stock_id"].astype(str) + df["type"].map(LISTED_TYPES)
    return df[["ticker", "industry_twse"]].reset_index(drop=True)


def attach_detailed_industry(panel: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Add `industry_twse` to the panel; missing detail falls back to `grp:<industry>`."""
    lookup = dict(zip(mapping["ticker"], mapping["industry_twse"]))
    detail = panel["ticker"].map(lookup)
    fallback = "grp:" + panel["industry"].astype(str)
    return panel.assign(industry_twse=detail.where(detail.notna(), fallback))


__all__ = ["detailed_industry_map", "attach_detailed_industry"]
