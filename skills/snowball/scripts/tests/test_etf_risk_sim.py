"""TDD tests for the ETF risk-control simulator: rule timing, no look-ahead, bands and costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import etf_risk_sim as rs

COSTS = {"fee_rate": 0.001425, "sell_tax_etf": 0.001, "slippage_each_side": 0.001}
TARGETS = {"tw": 0.45, "us": 0.40, "cash": 0.15}


def _prices(days, tw=None, us=None):
    tw = np.ones(len(days)) * 100 if tw is None else tw
    us = np.ones(len(days)) * 50 if us is None else us
    return pd.DataFrame({"tw": tw, "us": us}, index=days, dtype=float)


def test_us_sleeve_uses_strictly_earlier_us_close():
    tw_days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    us = pd.Series([10.0, 11.0, 12.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    out = rs.align_strictly_before(us, tw_days)
    assert out.tolist()[1:] == [10.0, 11.0]
    assert np.isnan(out.iloc[0])


def test_month_end_signal_days_and_next_day_execution():
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    sig = rs.month_end_days(days, start="2024-01-31")
    assert [str(d.date()) for d in sig] == ["2024-01-31", "2024-02-29", "2024-03-29"]
    assert rs.next_day(days, sig[0]) == pd.Timestamp("2024-02-01")


def test_static_rule_trades_only_in_june_december_and_only_off_band_sleeves():
    rule = rs.StaticRule(TARGETS, band=0.05)
    cur = {"tw": 0.52, "us": 0.38, "cash": 0.10}
    assert rule.decide(pd.Timestamp("2024-03-29"), cur, None) is None
    new = rule.decide(pd.Timestamp("2024-06-28"), cur, None)
    assert new["tw"] == pytest.approx(0.45) and new["us"] == pytest.approx(0.38)
    assert new["cash"] == pytest.approx(1 - 0.45 - 0.38)


def test_trend_rule_uses_only_data_up_to_signal_date():
    days = pd.bdate_range("2023-01-02", periods=260)
    tw = np.linspace(100, 130, 260)
    tw[-5:] = 50.0                       # crash AFTER the signal date below
    px = _prices(days, tw=tw)
    rule = rs.TrendRule(TARGETS, sma_days=200, band=0.05)
    signal = days[-6]
    new = rule.decide(signal, {"tw": 0.45, "us": 0.40, "cash": 0.15}, px)
    assert new is None                   # still above its SMA at the signal date: nothing to do
    crash_sig = days[-1]
    new = rule.decide(crash_sig, {"tw": 0.45, "us": 0.40, "cash": 0.15}, px)
    assert new["tw"] == 0.0 and new["cash"] == pytest.approx(0.60)


def test_vol_target_weight_formula():
    days = pd.bdate_range("2023-01-02", periods=100)
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, len(days))
    px = _prices(days, tw=100 * np.cumprod(1 + r), us=50 * np.cumprod(1 + r))
    rule = rs.VolTargetRule(TARGETS, vol_target=0.10, window=63, band=0.05)
    sigma = pd.Series(r, index=days).iloc[-63:].std() * np.sqrt(252)
    expect_r = min(0.85, 0.10 / sigma)
    new = rule.decide(days[-1], {"tw": 0.45, "us": 0.40, "cash": 0.15}, px)
    assert new["tw"] == pytest.approx(expect_r * 45 / 85, rel=1e-6)
    assert new["us"] == pytest.approx(expect_r * 40 / 85, rel=1e-6)


def test_simulation_applies_trade_next_day_and_charges_costs():
    days = pd.bdate_range("2024-01-01", "2024-02-29")
    px = _prices(days)
    res = rs.simulate(px, rs.StaticRule(TARGETS, band=0.05), signal_start="2024-01-31",
                      cash_yield=0.0, costs=COSTS, initial_nav=100000.0)
    first_exec = pd.Timestamp("2024-02-01")
    assert res.nav.index[0] == first_exec
    buys = 0.85 * 100000
    expected_cost = buys * (COSTS["fee_rate"] + COSTS["slippage_each_side"])
    assert res.costs_paid == pytest.approx(expected_cost, rel=1e-9)
    assert res.nav.iloc[-1] == pytest.approx(100000 - expected_cost, rel=1e-9)


def test_buy_and_hold_rule_never_trades_after_entry():
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    px = _prices(days, tw=np.linspace(100, 120, len(days)))
    res = rs.simulate(px, rs.HoldRule({"tw": 1.0, "us": 0.0, "cash": 0.0}), signal_start="2024-01-31",
                      cash_yield=0.0, costs=COSTS, initial_nav=100000.0)
    assert res.n_trade_days == 1
