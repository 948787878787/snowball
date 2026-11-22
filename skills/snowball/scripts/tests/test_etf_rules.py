"""TDD tests for build-in share sizing: whole odd lots for TW listings, fractional shares for USD ETFs."""

from __future__ import annotations

import pandas as pd
import pytest

import etf_rules as er

USD_CFG = {"initial_capital_twd": 100000, "targets": {"006208.TW": 0.45, "VOO": 0.40, "CASH": 0.15},
           "build_in": {"months": 3}}
START = pd.Timestamp("2026-12-01")


def test_usd_etf_keeps_fractional_shares_rounded_to_three_decimals():
    sug = er.build_in(USD_CFG, START, START, invested={}, price_twd={"006208.TW": 190.0, "VOO": 20000.0})
    by = {s["symbol"]: s for s in sug}
    assert by["VOO"]["shares"] == pytest.approx(round(40000 / 3 / 20000.0, 3))
    assert isinstance(by["VOO"]["shares"], float) and by["VOO"]["shares"] > 0


def test_tw_listing_stays_whole_odd_lot_integer():
    sug = er.build_in(USD_CFG, START, START, invested={}, price_twd={"006208.TW": 190.0, "VOO": 20000.0})
    tw = next(s for s in sug if s["symbol"] == "006208.TW")
    assert tw["shares"] == int(15000 // 190.0) and isinstance(tw["shares"], int)


def test_share_formatting_helper():
    assert er.format_shares(0.667) == "0.667"
    assert er.format_shares(78) == "78"
