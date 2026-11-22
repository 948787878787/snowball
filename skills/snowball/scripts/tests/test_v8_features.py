"""TDD tests for Snowball v8 factor dimensions: chip flows, risk/lottery, macro transmission."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import v8_features as vf

DAYS = pd.bdate_range("2024-01-01", periods=200)
WEEKS = pd.date_range("2024-02-02", DAYS[-1], freq="W-FRI")


def _wide(value, cols=("2330.TW",), index=DAYS) -> pd.DataFrame:
    return pd.DataFrame({c: value for c in cols}, index=index, dtype=float)


def test_asof_weekly_uses_last_valid_value_per_column():
    # US holiday on a Friday: the joined frame has a row (TW/FX traded) but SPX is NaN that day.
    days = pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03"])
    daily = pd.DataFrame({"usd": [32.0, 32.1, 32.2, 32.3, 32.4],
                          "spx": [5000.0, 5010.0, 5020.0, 5030.0, np.nan]}, index=days)
    out = vf.asof_weekly(daily, pd.DatetimeIndex([pd.Timestamp("2026-04-03")]), max_gap_days=7)
    assert out.loc["2026-04-03", "usd"] == 32.4
    assert out.loc["2026-04-03", "spx"] == 5030.0


def test_asof_weekly_masks_rows_older_than_gap():
    daily = _wide(1.0).iloc[:50]
    out = vf.asof_weekly(daily, WEEKS, max_gap_days=7)
    assert out.loc[WEEKS[0], "2330.TW"] == 1.0
    assert out.loc[WEEKS[-1]].isna().all()


def test_chip_flow_ratio_is_net_buy_over_volume():
    tidy = pd.DataFrame({"stock_id": "2330", "date": DAYS, "foreign_net": 100.0, "trust_net": -50.0})
    vol = _wide(1000.0)
    out = vf.chip_flow_features(tidy, vol, WEEKS, {"2330": "2330.TW"})
    row = out.loc[(out["date"] == WEEKS[-1]) & (out["ticker"] == "2330.TW")].iloc[0]
    assert row["chip_foreign_flow_20d"] == pytest.approx(0.1)
    assert row["chip_trust_flow_20d"] == pytest.approx(-0.05)


def test_margin_change_scaled_by_volume_and_holding_change_in_points():
    fridays = pd.date_range("2024-01-05", DAYS[-1], freq="W-FRI")
    margin = pd.DataFrame({"stock_id": "2330", "date": fridays, "margin_lots": np.arange(len(fridays), dtype=float)})
    hold = pd.DataFrame({"stock_id": "2330", "date": fridays, "foreign_hold_pct": np.arange(len(fridays), dtype=float) * 0.5})
    vol = _wide(1000.0)
    out = vf.chip_stock_features(margin, hold, vol, WEEKS, {"2330": "2330.TW"})
    row = out.loc[out["date"] == WEEKS[-1]].iloc[0]
    assert row["chip_margin_chg_4w"] == pytest.approx(4 * 1000 / (20 * 1000))  # +4 lots over 4 weeks
    assert row["chip_foreign_hold_chg_12w"] == pytest.approx(6.0)


def test_idio_vol_near_zero_when_stock_is_levered_market():
    rng = np.random.default_rng(0)
    mkt = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(DAYS))), index=DAYS)
    close = pd.DataFrame({"A.TW": mkt ** 2 / 100, "B.TW": 100 * np.cumprod(1 + rng.normal(0, 0.02, len(DAYS)))}, index=DAYS)
    out = vf.risk_features(close, close * 0 + 1e6, mkt, WEEKS).set_index(["date", "ticker"])
    last = WEEKS[-1]
    assert out.loc[(last, "A.TW"), "risk_idio_vol_60d"] < 0.25 * out.loc[(last, "B.TW"), "risk_idio_vol_60d"]


def test_max_daily_return_over_21_days():
    close = _wide(100.0)
    close.iloc[-5:, 0] = 110.0  # +10% jump 5 days before the end
    out = vf.risk_features(close, close * 0 + 1e6, close.iloc[:, 0] * 0 + 100, WEEKS)
    row = out.loc[out["date"] == WEEKS[-1]].iloc[0]
    assert row["risk_max_21d"] == pytest.approx(0.10)


def test_macro_beta_recovers_loading_and_signal_uses_recent_shock():
    weeks = pd.date_range("2020-01-03", periods=200, freq="W-FRI")
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"usd": rng.normal(0, 0.01, 200), "rate": rng.normal(0, 0.05, 200), "oil": rng.normal(0, 0.04, 200)}, index=weeks)
    excess = pd.DataFrame({"X.TW": 0.5 * f["usd"] - 0.2 * f["oil"]}, index=weeks)
    betas = vf.rolling_factor_betas(excess, f, window=104, min_obs=52)
    assert betas["usd"].iloc[-1, 0] == pytest.approx(0.5, abs=1e-6)
    assert betas["oil"].iloc[-1, 0] == pytest.approx(-0.2, abs=1e-6)
    assert betas["rate"].iloc[-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(betas["usd"].iloc[40, 0])  # fewer than min_obs observations
    sig = vf.macro_transmission_signal(betas, f, shock_weeks=4, window=104)
    expected = sum(betas[k].iloc[-1, 0] * f[k].iloc[-4:].sum() / (f[k].iloc[-104:].std() * 2.0) for k in f)
    assert sig.iloc[-1, 0] == pytest.approx(expected)


def test_monetary_rate_shock_kept_only_when_stocks_and_bonds_fall_together():
    weeks = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    dy = pd.Series(np.tile([0.1, -0.1], 15), index=weeks)
    eq_neg = -dy * 2    # yields up, equities down -> negative correlation -> monetary
    eq_pos = dy * 2
    assert vf.monetary_rate_shock(dy, eq_neg, window=13).iloc[-1] == pytest.approx(dy.iloc[-1])
    assert vf.monetary_rate_shock(dy, eq_pos, window=13).iloc[-1] == 0.0
