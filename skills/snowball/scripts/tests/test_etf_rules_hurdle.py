"""TDD tests for core ETF suggestion rules and the 6% non-cumulative HWM hurdle split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import etf_config as ec
import etf_hurdle as eh
import etf_rules as er

CFG = ec.load_config(Path(__file__).resolve().parents[4] / "configs" / "core_etf.json")
START = pd.Timestamp("2026-12-01")


def test_build_in_first_month_buys_one_third_of_each_risk_target():
    sug = er.build_in(CFG, START, pd.Timestamp("2026-12-01"), invested={}, price_twd={"006208.TW": 190.0, "00646.TW": 74.0})
    by = {s["symbol"]: s for s in sug}
    assert by["006208.TW"]["amount_twd"] == pytest.approx(45000 / 3)
    assert by["006208.TW"]["shares"] == int(15000 // 190.0)
    assert by["00646.TW"]["amount_twd"] == pytest.approx(40000 / 3)


def test_build_in_second_month_tops_up_to_two_thirds_and_stops_after_three():
    second = er.build_in(CFG, START, pd.Timestamp("2027-01-05"), invested={"006208.TW": 15000.0, "00646.TW": 13000.0},
                         price_twd={"006208.TW": 190.0, "00646.TW": 74.0})
    by = {s["symbol"]: s["amount_twd"] for s in second}
    assert by["006208.TW"] == pytest.approx(30000 - 15000)
    assert by["00646.TW"] == pytest.approx(40000 * 2 / 3 - 13000)
    assert er.build_in(CFG, START, pd.Timestamp("2027-03-02"), invested={}, price_twd={"006208.TW": 1, "00646.TW": 1}) == []


def test_rebalance_only_on_six_month_checks_and_outside_band():
    w_ok = {"006208.TW": 0.48, "00646.TW": 0.38, "CASH": 0.14}
    w_bad = {"006208.TW": 0.52, "00646.TW": 0.34, "CASH": 0.14}
    assert er.rebalance(CFG, START, pd.Timestamp("2027-04-01"), w_bad, nav=100000)["due"] is False
    ok = er.rebalance(CFG, START, pd.Timestamp("2027-06-01"), w_ok, nav=100000)
    assert ok["due"] is True and ok["trades"] == []
    bad = er.rebalance(CFG, START, pd.Timestamp("2027-06-01"), w_bad, nav=100000)
    trades = {t["symbol"]: t["amount_twd"] for t in bad["trades"]}
    assert trades == {"006208.TW": pytest.approx(-7000.0), "00646.TW": pytest.approx(6000.0)}


def test_drawdown_trigger_releases_buffer_in_thirds():
    idx = pd.Series([100.0, 90.0, 84.0], index=pd.to_datetime(["2027-01-04", "2027-02-01", "2027-03-01"]))
    none = er.drawdown_deploy(CFG, idx.iloc[:2], cash=15000, nav=100000)
    assert none["triggered"] == 0 and none["deploy_twd"] == 0
    one = er.drawdown_deploy(CFG, idx, cash=15000, nav=100000)
    assert one["drawdown"] == pytest.approx(-0.16)
    assert one["triggered"] == 1 and one["deploy_twd"] == pytest.approx(5000.0)
    split = {s: a for s, a in one["split"].items()}
    assert split["006208.TW"] == pytest.approx(5000 * 45 / 85)


def _nav(points):
    return pd.Series({pd.Timestamp(d): v for d, v in points}).sort_index()


def test_hurdle_split_matches_contract_over_three_years():
    nav = _nav([("2026-12-01", 100000), ("2027-11-30", 110000),
                ("2027-12-01", 106000), ("2028-11-30", 95400),
                ("2029-11-30", 114480)])
    flows = pd.Series({pd.Timestamp("2026-12-01"): 100000.0})
    years = eh.contract_split(nav, flows, start=START, as_of=pd.Timestamp("2029-11-30"), hurdle=0.06)
    assert [round(y["carry"], 6) for y in years] == [4000.0, 0.0, 8480.0]
    assert years[0]["investor_gain"] == pytest.approx(6000.0)
    assert years[1]["hwm_after"] == pytest.approx(106000.0)
    assert all(y["complete"] for y in years)


def test_partial_year_is_provisional_with_full_year_hurdle():
    nav = _nav([("2026-12-01", 100000), ("2027-05-31", 108000)])
    flows = pd.Series({pd.Timestamp("2026-12-01"): 100000.0})
    cur = eh.contract_split(nav, flows, start=START, as_of=pd.Timestamp("2027-05-31"), hurdle=0.06)[-1]
    assert cur["complete"] is False
    assert cur["carry"] == pytest.approx(2000.0) and cur["investor_gain"] == pytest.approx(6000.0)
