"""TDD tests for v8 investor-facing metrics: time under water and the 6% HWM hurdle carry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import v8_eval_metrics as em


def _weekly(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2022-01-07", periods=len(values), freq="W-FRI"), dtype=float)


def test_time_under_water_counts_longest_and_current_spell():
    nav = _weekly([1.0, 1.1, 1.0, 0.9, 1.05, 1.2, 1.1, 1.15])
    out = em.time_under_water(nav)
    assert out["max_weeks_under_water"] == 3      # weeks at 1.0, 0.9, 1.05 below the 1.1 peak
    assert out["current_weeks_under_water"] == 2  # 1.1 and 1.15 below the 1.2 peak
    assert out["recovered"] is False


def test_time_under_water_zero_for_monotone_curve():
    out = em.time_under_water(_weekly([1.0, 1.01, 1.02]))
    assert out["max_weeks_under_water"] == 0 and out["recovered"] is True


def test_hurdle_year_pays_only_above_hwm_and_six_percent():
    # Year 1 +10%: carry = 110 - max(100 HWM, 106) = 4 -> investor keeps 106, HWM 106.
    # Year 2 -10% from 106: no carry, HWM stays 106.
    # Year 3 +20% from 95.4 -> 114.48: hurdle 95.4*1.06 = 101.12 < HWM 106 -> carry 8.48.
    yearly = [0.10, -0.10, 0.20]
    out = em.hwm_hurdle_path(yearly, hurdle=0.06)
    assert out["carry"] == pytest.approx([4.0, 0.0, 8.48], abs=1e-9)
    assert out["investor_nav"][-1] == pytest.approx(106.0)


def test_rolling_three_year_carry_probability_on_weekly_returns():
    rets = _weekly(np.full(52 * 4, (1.10) ** (1 / 52) - 1))   # steady +10%/yr for 4 years
    out = em.rolling_hurdle_windows(rets, hurdle=0.06, years=3)
    assert out["n_windows"] == 52 + 1
    assert out["p_any_carry"] == pytest.approx(1.0)
    assert out["p_investor_hit_hurdle_every_year"] == pytest.approx(1.0)
    flat = em.rolling_hurdle_windows(_weekly(np.zeros(52 * 3)), hurdle=0.06, years=3)
    assert flat["p_any_carry"] == 0.0 and flat["mean_manager_share_of_profit"] == 0.0
