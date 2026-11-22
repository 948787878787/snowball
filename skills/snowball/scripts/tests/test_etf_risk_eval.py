"""TDD tests for ETF risk-control metrics and the pre-registered gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import etf_risk_eval as ee


def _nav(values, start="2014-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_performance_metrics_on_known_path():
    nav = _nav(np.concatenate([np.linspace(100, 120, 300), np.linspace(120, 90, 100), np.linspace(90, 130, 357)]))
    m = ee.performance(nav)
    assert m["mdd"] == pytest.approx(90 / 120 - 1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    assert m["cagr"] == pytest.approx((130 / 100) ** (1 / years) - 1)
    assert m["calmar"] == pytest.approx(m["cagr"] / abs(m["mdd"]))
    assert m["max_weeks_under_water"] > 0


def test_full_calendar_years_exclude_partial_first_and_last():
    nav = _nav(np.linspace(100, 200, 1000), start="2013-08-01")
    yr = ee.calendar_returns(nav)
    assert yr.index.min() == 2014 and yr.index.max() == nav.index[-1].year - 1


def test_gate_requires_all_four_conditions():
    s0 = {"full": {"mdd": -0.25, "cagr": 0.10, "calmar": 0.40, "max_weeks_under_water": 70},
          "h1": {"calmar": 0.5}, "h2": {"calmar": 0.3}}
    good = {"full": {"mdd": -0.18, "cagr": 0.09, "calmar": 0.50, "max_weeks_under_water": 60},
            "h1": {"calmar": 0.6}, "h2": {"calmar": 0.35}}
    g = ee.gate(good, s0)
    assert g["pass"] is True
    shallow_only = {**good, "full": {**good["full"], "mdd": -0.21}}
    assert ee.gate(shallow_only, s0)["mdd_5pp_shallower"] is False
    slow = {**good, "full": {**good["full"], "cagr": 0.08}}
    assert ee.gate(slow, s0)["cagr_within_1p5pp"] is False
    weak_half = {**good, "h2": {"calmar": 0.29}}
    assert ee.gate(weak_half, s0)["pass"] is False


def test_pick_forward_arm_prefers_higher_calmar_among_passers():
    results = {"A1_trend": {"gate": {"pass": True}, "full": {"calmar": 0.6}},
               "A2_voltarget": {"gate": {"pass": True}, "full": {"calmar": 0.7}}}
    assert ee.pick_forward(results) == "A2_voltarget"
    results["A2_voltarget"]["gate"]["pass"] = False
    assert ee.pick_forward(results) == "A1_trend"
    results["A1_trend"]["gate"]["pass"] = False
    assert ee.pick_forward(results) is None
