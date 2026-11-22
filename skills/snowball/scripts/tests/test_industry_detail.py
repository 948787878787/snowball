"""TDD tests for the detailed TWSE/TPEx industry classification used by v8 neutralisation and caps."""

from __future__ import annotations

import pandas as pd

import industry_detail as ind


def _info() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_id": ["2330", "3105", "6488", "8299", "1101", "2330", "4142", "9999"],
        "type": ["twse", "tpex", "tpex", "tpex", "twse", "twse", "twse", "emerging"],
        "industry_category": ["半導體業", "半導體業", "其他電子類", "電子工業", "水泥工業", "電子工業",
                              "化學生技醫療", "半導體業"],
    })


def test_tpex_and_twse_name_variants_are_unified():
    out = ind.detailed_industry_map(_info())
    by = dict(zip(out["ticker"], out["industry_twse"]))
    assert by["2330.TW"] == "半導體業"          # specific category wins over the generic duplicate row
    assert by["3105.TWO"] == "半導體業"
    assert by["6488.TWO"] == "其他電子業"       # TPEx "...類" spelling folded into the TWSE name
    assert by["4142.TW"] == "生技醫療業"


def test_generic_only_category_is_left_missing_for_group_fallback():
    out = ind.detailed_industry_map(_info())
    by = dict(zip(out["ticker"], out["industry_twse"]))
    assert pd.isna(by["8299.TWO"])


def test_non_listed_types_and_bad_codes_are_dropped():
    out = ind.detailed_industry_map(_info())
    assert "9999.TW" not in set(out["ticker"]) and "9999.TWO" not in set(out["ticker"])


def test_attach_falls_back_to_coarse_group_when_detail_missing():
    panel = pd.DataFrame({"ticker": ["2330.TW", "8299.TWO", "1234.TW"],
                          "industry": ["semiconductors_hardware", "semiconductors_hardware", "other"]})
    mapping = pd.DataFrame({"ticker": ["2330.TW", "8299.TWO"], "industry_twse": ["半導體業", None]})
    out = ind.attach_detailed_industry(panel, mapping)
    assert out["industry_twse"].tolist() == ["半導體業", "grp:semiconductors_hardware", "grp:other"]
    assert len(out) == len(panel)
