"""TDD tests for TWSE/TPEx bulk daily chip-data parsers (institutional flows, margin, foreign holding)."""

from __future__ import annotations

import pandas as pd
import pytest

import tw_bulk_fetch as tbf

TWSE_T86 = {
    "stat": "OK",
    "fields": ["證券代號", "證券名稱"] + [f"c{i}" for i in range(2, 19)],
    "data": [
        ["2330", "台積電", "1,000", "400", "600", "0", "0", "0", "50", "10", "40", "-5"] + ["0"] * 7,
        ["0050", "元大台灣50", "1", "0", "1", "0", "0", "0", "0", "0", "0", "0"] + ["0"] * 7,
    ],
}
TPEX_INSTI = {
    "stat": "ok",
    "tables": [{
        "fields": ["代號", "名稱"] + [f"c{i}" for i in range(2, 24)],
        "data": [["6488", "環球晶", "900", "1,200", "-300"] + ["0"] * 8 + ["25"] + ["0"] * 10],
    }],
}
TWSE_MARGIN = {
    "stat": "OK",
    "tables": [
        {"fields": ["項目"], "data": [["融資(交易單位)", "1"]]},
        {"fields": ["代號", "名稱"] + [f"c{i}" for i in range(2, 16)],
         "data": [["1101", "台泥", "102", "148", "1", "11,655", "11,608", "0", "0", "0", "0", "0", "0", "0", "0", " "]]},
    ],
}
TPEX_MARGIN = {
    "stat": "ok",
    "tables": [{"fields": ["代號", "名稱"] + [f"c{i}" for i in range(2, 20)],
                "data": [["8069", "元太", "5,000", "10", "20", "0", "4,990"] + ["0"] * 13]}],
}
TWSE_QFII = {
    "stat": "OK",
    "fields": ["證券代號", "證券名稱"] + [f"c{i}" for i in range(2, 12)],
    "data": [["2330", "台積電", "X", "100", "0", "70", "0", 72.5, "100", "100", "", 0]],
}
TPEX_QFII = {
    "stat": "ok",
    "tables": [{"fields": ["排行", "代號", "名稱"] + [f"c{i}" for i in range(3, 10)],
                "data": [["1", "8455", "大拓-KY", "25", "3", "22", "12.19%", "87.81%", "100%", ""]]}],
}

DAY = pd.Timestamp("2025-11-14")


def test_parse_twse_institutional_keeps_common_stocks_and_net_columns():
    out = tbf.parse_institutional(TWSE_T86, "twse", DAY)
    assert list(out["stock_id"]) == ["2330"]
    row = out.iloc[0]
    assert row["foreign_net"] == 600 and row["trust_net"] == 40
    assert row["date"] == DAY


def test_parse_tpex_institutional_uses_foreign_ex_dealer_and_trust_columns():
    row = tbf.parse_institutional(TPEX_INSTI, "tpex", DAY).iloc[0]
    assert row["stock_id"] == "6488"
    assert row["foreign_net"] == -300 and row["trust_net"] == 25


def test_parse_margin_balance_in_lots_from_stock_table():
    tw = tbf.parse_margin(TWSE_MARGIN, "twse", DAY).iloc[0]
    tp = tbf.parse_margin(TPEX_MARGIN, "tpex", DAY).iloc[0]
    assert (tw["stock_id"], tw["margin_lots"]) == ("1101", 11608)
    assert (tp["stock_id"], tp["margin_lots"]) == ("8069", 4990)


def test_stock_table_detected_even_when_first_rows_are_bond_etfs():
    etf_rows = [[f"00{700 + i}B", "ETF", "1", "0", "0", "0", "1"] + ["0"] * 13 for i in range(80)]
    payload = {"stat": "ok", "tables": [{"data": etf_rows + TPEX_MARGIN["tables"][0]["data"]}]}
    out = tbf.parse_margin(payload, "tpex", DAY)
    assert list(out["stock_id"]) == ["8069"]


def test_parse_foreign_holding_ratio_handles_float_and_percent_string():
    tw = tbf.parse_foreign_holding(TWSE_QFII, "twse", DAY).iloc[0]
    tp = tbf.parse_foreign_holding(TPEX_QFII, "tpex", DAY).iloc[0]
    assert tw["foreign_hold_pct"] == pytest.approx(72.5)
    assert tp["stock_id"] == "8455" and tp["foreign_hold_pct"] == pytest.approx(87.81)


def test_non_trading_day_payload_returns_empty_frame():
    out = tbf.parse_institutional({"stat": "很抱歉，沒有符合條件的資料!"}, "twse", DAY)
    assert out.empty and {"stock_id", "date", "foreign_net", "trust_net"} <= set(out.columns)


def test_weekly_snapshot_dates_are_fridays_in_range():
    days = tbf.weekly_fridays("2025-11-01", "2025-11-30")
    assert all(d.weekday() == 4 for d in days)
    assert days[0] == pd.Timestamp("2025-11-07") and len(days) == 4


def test_url_builders_use_roc_free_formats():
    assert "date=20251114" in tbf.build_url("institutional", "twse", DAY)
    assert "date=2025%2F11%2F14" in tbf.build_url("institutional", "tpex", DAY)
