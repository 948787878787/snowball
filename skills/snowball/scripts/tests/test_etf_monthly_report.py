"""TDD tests for the one-click monthly investor report (pure parts; prices injected, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import etf_config as ec
import etf_ledger as el
import etf_monthly_report as mr

CFG = ec.load_config(Path(__file__).resolve().parents[4] / "configs" / "core_etf.json")
DAYS = pd.bdate_range("2026-11-20", "2027-01-29")


def _ledger(tmp_path):
    rows = [["2026-12-01", "deposit", "", "", "", "", "", 100000, ""],
            ["2026-12-01", "buy", "006208.TW", 78, 190.0, 21, 0, "", ""],
            ["2026-12-01", "buy", "00646.TW", 180, 74.0, 19, 0, "", ""],
            ["2027-01-05", "buy", "006208.TW", 78, 195.0, 22, 0, "", ""],
            ["2027-01-20", "dividend", "006208.TW", "", "", "", "", 150, ""]]
    p = tmp_path / "l.csv"
    pd.DataFrame(rows, columns=el.COLUMNS).to_csv(p, index=False)
    return el.load_ledger(p)


def _market():
    rng = np.random.default_rng(5)
    prices = pd.DataFrame({"006208.TW": 190 * np.cumprod(1 + rng.normal(0.001, 0.01, len(DAYS))),
                           "00646.TW": 74 * np.cumprod(1 + rng.normal(0.001, 0.01, len(DAYS)))}, index=DAYS)
    bench = pd.Series(84 * np.cumprod(1 + rng.normal(0.001, 0.01, len(DAYS))), index=DAYS)
    return prices, pd.Series(32.0, index=DAYS), bench


def test_previous_month_default():
    assert mr.previous_month(pd.Timestamp("2027-02-01")) == "2027-01"
    assert mr.previous_month(pd.Timestamp("2027-01-01")) == "2026-12"


def test_net_invested_per_symbol(tmp_path):
    inv = mr.net_invested(_ledger(tmp_path), pd.Timestamp("2027-01-31"))
    assert inv["006208.TW"] == pytest.approx(78 * 190 + 21 + 78 * 195 + 22)


def test_report_has_five_to_eight_lines_with_key_numbers(tmp_path):
    led = _ledger(tmp_path)
    prices, fx, bench = _market()
    data = mr.gather(CFG, led, prices, fx, bench, "2027-01")
    lines = mr.render(data, CFG)
    assert 5 <= len(lines) <= 8
    text = "\n".join(lines)
    assert "2027/01" in text and "總淨值" in text and "0050" in text
    assert "6%" in text and "出資方" in text and "操作方" in text
    assert "最大回撤" in text and "水下" in text
    assert "買 006208" in text and "配息 150" in text
    assert data["nav"] == pytest.approx(mr.gather(CFG, led, prices, fx, bench, "2027-01")["nav"])


def test_report_says_no_trade_when_nothing_happened(tmp_path):
    led = _ledger(tmp_path)
    prices, fx, bench = _market()
    lines = mr.render(mr.gather(CFG, led, prices, fx, bench, "2026-12"), CFG)
    assert any("建倉第 2/3 批" in ln for ln in lines)


def test_drawdown_tip_is_suppressed_while_build_in_is_running(tmp_path):
    led = _ledger(tmp_path)
    prices, fx, bench = _market()
    crash = bench.copy()
    crash.loc["2026-12-20":] = crash.loc["2026-12-20":] * 0.7
    tips = mr.gather(CFG, led, prices, fx, crash, "2026-12")["tips"]
    assert any("建倉" in t for t in tips)
    assert not any("現金緩衝" in t for t in tips)
    after = mr.gather(CFG, led, prices, fx, crash, "2027-01")["tips"]
    assert not any("現金緩衝" in t for t in after)   # January report -> next month is still tranche 3
