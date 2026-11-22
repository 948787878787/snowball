"""TDD tests for MOPS quarterly financial parsing, PIT availability and quality factors."""

from __future__ import annotations

import pandas as pd
import pytest

import tw_mops_financials as tmf

INCOME_HTML = """
<table><tr><th>公司 代號</th><th>公司名稱</th><th>營業收入</th><th>營業成本</th><th>營業毛利（毛損）淨額</th>
<th>淨利（淨損）歸屬於母公司業主</th></tr>
<tr><td>2330</td><td>台積電</td><td>1,000</td><td>400</td><td>600</td><td>300</td></tr>
<tr><td>6488</td><td>環球晶</td><td>500</td><td>350</td><td>150</td><td>--</td></tr></table>
<table><tr><th>公司 代號</th><th>公司名稱</th><th>利息淨收益</th><th>淨利（淨損）歸屬於母公司業主</th></tr>
<tr><td>2881</td><td>富邦金</td><td>90</td><td>50</td></tr></table>
"""
BALANCE_HTML = """
<table><tr><th>公司 代號</th><th>公司名稱</th><th>資產總計</th><th>歸屬於母公司業主之權益合計</th></tr>
<tr><td>2330</td><td>台積電</td><td>5,000</td><td>3,000</td></tr></table>
"""


def test_parse_income_maps_column_variants_and_keeps_financials():
    out = tmf.parse_statement(INCOME_HTML, "income").set_index("stock_id")
    assert out.loc["2330", "revenue"] == 1000 and out.loc["2330", "gross_profit"] == 600
    assert out.loc["2330", "net_income"] == 300
    assert pd.isna(out.loc["6488", "net_income"])
    assert "2881" in out.index and pd.isna(out.loc["2881", "gross_profit"])


def test_parse_balance_total_assets_and_equity():
    row = tmf.parse_statement(BALANCE_HTML, "balance").set_index("stock_id").loc["2330"]
    assert row["total_assets"] == 5000 and row["equity"] == 3000


@pytest.mark.parametrize("year,season,expected", [
    (2024, 1, "2024-05-16"), (2024, 2, "2024-08-15"), (2024, 3, "2024-11-15"), (2024, 4, "2025-04-01"),
])
def test_available_date_is_day_after_statutory_deadline(year, season, expected):
    assert tmf.available_date(year, season) == pd.Timestamp(expected)


def _quarters(stock: str, gm: list, ni: list, assets: list) -> pd.DataFrame:
    rows = []
    for i, (g, n, a) in enumerate(zip(gm, ni, assets)):
        year, season = 2022 + i // 4, i % 4 + 1
        rows.append({"stock_id": stock, "year": year, "season": season, "revenue": 100.0,
                     "gross_profit": g, "net_income": n, "total_assets": a, "equity": a / 2})
    return pd.DataFrame(rows)


def test_quality_factors_are_year_over_year_same_season():
    q = _quarters("1101", gm=[20, 20, 20, 20, 30, 30, 30, 30], ni=[5] * 8, assets=[100] * 4 + [125] * 4)
    out = tmf.quality_factors(q).set_index(["year", "season"])
    row = out.loc[(2023, 2)]
    assert row["q_gm_yoy"] == pytest.approx(0.10)          # 30% - 20% margin
    assert row["q_asset_growth"] == pytest.approx(0.25)
    assert row["q_roa_yoy"] == pytest.approx(5 / 125 - 5 / 100)
    assert pd.isna(out.loc[(2022, 4), "q_gm_yoy"])          # no prior year yet
    assert row["available"] == tmf.available_date(2023, 2)


def test_quality_factors_asof_merge_uses_only_released_statements():
    q = tmf.quality_factors(_quarters("1101", [20] * 4 + [30] * 4, [5] * 8, [100] * 8))
    weeks = pd.DataFrame({"date": pd.to_datetime(["2023-05-12", "2023-05-19"]), "stock_id": "1101"})
    merged = tmf.merge_asof_quality(weeks, q, max_staleness_days=200).set_index("date")
    # 2023Q1 becomes available 2023-05-16: the 2023-05-12 row must still use 2022Q4 (no YoY -> NaN)
    assert pd.isna(merged.loc["2023-05-12", "q_gm_yoy"])
    assert merged.loc["2023-05-19", "q_gm_yoy"] == pytest.approx(0.10)
