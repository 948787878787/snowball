"""TDD tests for ETF risk-control research data preparation (no strategy returns involved)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import etf_risk_data as rd

DAYS = pd.bdate_range("2014-12-24", periods=8)


def test_one_day_fx_spike_that_reverses_is_replaced_by_previous_value():
    fx = pd.Series([30.0, 30.1, 33.5, 30.2, 30.3, 30.2, 30.1, 30.0], index=DAYS)
    out = rd.clean_reversing_spikes(fx, threshold=0.05)
    assert out.iloc[2] == pytest.approx(30.1)
    assert out.drop(DAYS[2]).equals(fx.drop(DAYS[2]))


def test_genuine_level_shift_is_kept():
    fx = pd.Series([30.0, 30.0, 32.0, 32.1, 32.0, 32.2, 32.1, 32.0], index=DAYS)
    assert rd.clean_reversing_spikes(fx, threshold=0.05).equals(fx)


def test_us_sleeve_in_twd_compounds_usd_return_and_fx_change():
    usd = pd.Series([100.0, 110.0], index=DAYS[:2])
    fx = pd.Series([30.0, 33.0], index=DAYS[:2])
    twd = rd.us_sleeve_twd(usd, fx, annual_fee=0.0)
    assert twd.iloc[1] / twd.iloc[0] == pytest.approx(1.10 * 1.10)


def test_fee_drag_is_daily_compounded():
    days = pd.bdate_range("2020-01-01", periods=253)
    usd = pd.Series(100.0, index=days)
    fx = pd.Series(30.0, index=days)
    twd = rd.us_sleeve_twd(usd, fx, annual_fee=0.0040)
    assert twd.iloc[-1] / twd.iloc[0] == pytest.approx(1 - 0.0040, abs=2e-4)


def test_tracking_difference_is_annualised_log_gap():
    days = pd.bdate_range("2020-01-01", periods=253)
    a = pd.Series(np.linspace(100, 110, 253), index=days)
    b = a * np.exp(-0.01 * np.arange(253) / 252)
    assert rd.tracking_difference(b, a) == pytest.approx(-0.01, abs=1e-3)


def test_default_threshold_keeps_large_genuine_fx_moves():
    # 2025-05: TWD rallied 5% in a day and partly reversed; that is market data, not a bad tick.
    days = pd.bdate_range("2025-05-01", periods=4)
    fx = pd.Series([32.11, 30.70, 29.16, 29.98], index=days)
    assert rd.clean_reversing_spikes(fx).equals(fx)
    bad = pd.Series([30.02, 30.09, 1.80, 30.11], index=days)
    assert rd.clean_reversing_spikes(bad).iloc[2] == pytest.approx(30.09)


def test_repo_etf_risk_config_is_draft_with_two_candidates_and_gate():
    import json
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parents[4] / "configs" / "etf_risk_v1.json").read_text(encoding="utf-8"))
    assert cfg["status"] in {"DRAFT", "FROZEN"}
    assert sum(cfg["targets"].values()) == pytest.approx(1.0)
    assert cfg["gate"]["applies_to"] == ["A1_trend", "A2_voltarget"]
    assert cfg["data"]["fx_spike_threshold"] == 0.20
