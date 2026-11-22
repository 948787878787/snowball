"""TDD tests for core ETF NAV, time-weighted returns and drawdown metrics."""

from __future__ import annotations

import pandas as pd
import pytest

import etf_ledger as el
import etf_portfolio as ep

DAYS = pd.bdate_range("2026-12-01", "2027-01-29")


def _ledger(tmp_path, rows):
    p = tmp_path / "l.csv"
    pd.DataFrame(rows, columns=el.COLUMNS).to_csv(p, index=False)
    return el.load_ledger(p)


def _prices(a=100.0, voo=500.0):
    return pd.DataFrame({"A.TW": a, "VOO": voo}, index=DAYS, dtype=float)


def test_nav_converts_usd_holdings_with_fx(tmp_path):
    led = _ledger(tmp_path, [["2026-12-01", "deposit", "", "", "", "", "", 100000, ""],
                             ["2026-12-01", "buy", "A.TW", 100, 100.0, 0, 0, "", ""],
                             ["2026-12-01", "buy", "VOO", 2, 500.0, 0, 0, 32000, ""]])
    fx = pd.Series(32.0, index=DAYS)
    nav = ep.nav_series(led, _prices(), fx, DAYS[-1])
    assert nav.iloc[0] == pytest.approx(100000 - 10000 - 32000 + 100 * 100 + 2 * 500 * 32)


def test_twr_ignores_deposits(tmp_path):
    led = _ledger(tmp_path, [["2026-12-01", "deposit", "", "", "", "", "", 50000, ""],
                             ["2026-12-15", "deposit", "", "", "", "", "", 50000, ""]])
    nav = ep.nav_series(led, _prices(), pd.Series(32.0, index=DAYS), DAYS[-1])
    idx = ep.twr_index(nav, el.external_flows(led))
    assert idx.iloc[-1] == pytest.approx(1.0)


def test_period_returns_and_drawdown(tmp_path):
    px = _prices()
    px.loc["2026-12-15":, "A.TW"] = 110.0
    px.loc["2027-01-04":, "A.TW"] = 99.0
    led = _ledger(tmp_path, [["2026-12-01", "deposit", "", "", "", "", "", 10000, ""],
                             ["2026-12-01", "buy", "A.TW", 100, 100.0, 0, 0, "", ""]])
    nav = ep.nav_series(led, px, pd.Series(32.0, index=DAYS), DAYS[-1])
    idx = ep.twr_index(nav, el.external_flows(led))
    assert ep.period_return(idx, "2026-12-01", "2026-12-31") == pytest.approx(0.10)
    assert ep.period_return(idx, "2027-01-01", "2027-01-29") == pytest.approx(99 / 110 - 1)
    m = ep.risk_metrics(idx)
    assert m["mdd"] == pytest.approx(99 / 110 - 1)
    assert m["current_weeks_under_water"] >= 3


def test_benchmark_total_return_over_period():
    adj = pd.Series([100.0, 105.0, 110.0], index=pd.to_datetime(["2026-11-30", "2026-12-15", "2026-12-31"]))
    assert ep.benchmark_return(adj, "2026-12-01", "2026-12-31") == pytest.approx(0.10)
