"""TDD tests for the weekly entry-point check (5-8 lines, suggestions only, no timing rules)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import etf_config as ec
import etf_entry_check as ck
import etf_ledger as el

CFG = ec.load_config(Path(__file__).resolve().parents[4] / "configs" / "core_etf.json")
DAYS = pd.bdate_range("2024-01-01", "2027-06-30")


def _market(bench_last_drop=0.0):
    prices = pd.DataFrame({"006208.TW": 190.0, "00646.TW": 74.0}, index=DAYS)
    bench = pd.Series(100.0, index=DAYS)
    bench.iloc[-1] = 100.0 * (1 - bench_last_drop)
    return prices, pd.Series(dtype=float), bench


def _ledger(tmp_path, rows):
    p = tmp_path / "l.csv"
    pd.DataFrame(rows, columns=el.COLUMNS).to_csv(p, index=False)
    return el.load_ledger(p)


FULL = [["2026-12-01", "deposit", "", "", "", "", "", 100000, ""],
        ["2026-12-01", "buy", "006208.TW", 236, 190.0, 64, 0, "", ""],
        ["2026-12-01", "buy", "00646.TW", 540, 74.0, 57, 0, "", ""]]


def test_empty_ledger_shows_first_tranche_and_discipline_line():
    prices, fx, bench = _market()
    lines = ck.entry_check(CFG, None, prices, fx, bench, pd.Timestamp("2027-06-30"))
    text = "\n".join(lines)
    assert 5 <= len(lines) <= 8
    assert "建倉第 1/3 批" in text and "006208" in text and "00646" in text
    assert "限價" in text and "溢價" in text and "不下單" in text


def test_drawdown_line_reports_gap_to_next_trigger():
    prices, fx, bench = _market(bench_last_drop=0.12)
    lines = ck.entry_check(CFG, None, prices, fx, bench, pd.Timestamp("2027-06-30"))
    assert any("-12.0%" in ln and "-15%" in ln and "3.0 個百分點" in ln for ln in lines)


def test_triggered_drawdown_after_build_in_lists_release_and_split(tmp_path):
    led = _ledger(tmp_path, FULL)
    prices, fx, bench = _market(bench_last_drop=0.16)
    lines = ck.entry_check(CFG, led, prices, fx, bench, pd.Timestamp("2027-06-30"))
    line = next(ln for ln in lines if ln.startswith("回撤："))
    assert "可釋放" in line and "006208" in line and "00646" in line


def test_idle_cash_suggests_buy_only_top_up_of_underweight_line(tmp_path):
    rows = FULL + [["2027-03-20", "deposit", "", "", "", "", "", 20000, "new cash"]]
    led = _ledger(tmp_path, rows)
    prices, fx, bench = _market()
    lines = ck.entry_check(CFG, led, prices, fx, bench, pd.Timestamp("2027-04-15"))
    line = next(ln for ln in lines if "閒置現金" in ln)
    assert "只買不賣" in line


def test_no_timing_rules_are_used():
    src = (Path(ck.__file__)).read_text(encoding="utf-8")
    assert "rolling(" not in src and "sma" not in src.lower() and "vol_target" not in src


def test_rebalance_line_during_build_in_does_not_report_deviations():
    prices, fx, bench = _market()
    lines = ck.entry_check(CFG, None, prices, fx, bench, pd.Timestamp("2027-06-30"))
    line = next(ln for ln in lines if ln.startswith("再平衡"))
    assert "建倉完成後" in line and "偏離" not in line
