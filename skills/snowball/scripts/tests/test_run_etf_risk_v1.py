"""TDD tests for the one-shot ETF risk-control runner (guard and synthetic end-to-end)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_etf_risk_v1 as rr

CFG = json.loads((Path(__file__).resolve().parents[4] / "configs" / "etf_risk_v1.json").read_text(encoding="utf-8"))


def test_runner_refuses_unless_frozen_with_commit():
    with pytest.raises(PermissionError):
        rr.require_frozen({**CFG, "status": "DRAFT", "code_commit": "abc"})
    with pytest.raises(PermissionError):
        rr.require_frozen({**CFG, "status": "FROZEN", "code_commit": None})
    rr.require_frozen({**CFG, "status": "FROZEN", "code_commit": "abc"})


def test_end_to_end_on_synthetic_prices(tmp_path):
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2012-06-22", "2026-11-20")
    tw = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, len(days))), index=days)
    us_days = pd.bdate_range("2012-06-01", "2026-11-20")
    us = pd.Series(900 * np.cumprod(1 + rng.normal(0.0004, 0.011, len(us_days))), index=us_days)
    prices = rr.build_prices(tw, us)
    out = rr.evaluate_arms(prices, CFG)
    day252 = prices.index[251]
    month_ends = prices.index.to_series().groupby([prices.index.year, prices.index.month]).max()
    assert out["first_signal"] == str(month_ends[month_ends >= day252].iloc[0].date())
    assert set(out["arms"]) == {"S0_static", "A1_trend", "A2_voltarget", "B_tw50_hold"}
    for k in ("A1_trend", "A2_voltarget"):
        assert set(out["arms"][k]["gate"]) >= {"mdd_5pp_shallower", "pass"}
    log = tmp_path / "trial_log.csv"
    pd.DataFrame([{"trial_id": 73, "stage": "seed"}]).to_csv(log, index=False)
    rr.log_trials(log, out, "abcdef0")
    assert pd.read_csv(log)["trial_id"].max() == 77
