"""TDD tests for the core ETF tracker config validation and the local ledger."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import etf_config as ec
import etf_ledger as el

REPO_CFG = Path(__file__).resolve().parents[4] / "configs" / "core_etf.json"


def test_repo_config_has_default_45_40_15_targets():
    cfg = ec.load_config(REPO_CFG)
    assert cfg["targets"] == {"006208.TW": 0.45, "00646.TW": 0.40, "CASH": 0.15}
    assert cfg["contract"]["hurdle_annual"] == 0.06


def test_config_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError, match="sum"):
        ec.validate({**ec.load_config(REPO_CFG), "targets": {"0050.TW": 0.5, "CASH": 0.2}})


def test_config_requires_cash_line():
    with pytest.raises(ValueError, match="CASH"):
        ec.validate({**ec.load_config(REPO_CFG), "targets": {"0050.TW": 1.0}})


def test_usd_symbol_detection():
    assert ec.is_usd("VOO") and not ec.is_usd("00646.TW") and not ec.is_usd("6488.TWO")


def _write(tmp_path: Path, rows) -> Path:
    p = tmp_path / "ledger.csv"
    pd.DataFrame(rows, columns=el.COLUMNS).to_csv(p, index=False)
    return p


def _rows():
    return [
        ["2026-12-01", "deposit", "", "", "", "", "", 100000, "initial"],
        ["2026-12-02", "buy", "006208.TW", 79, 190.0, 21, 0, "", ""],
        ["2026-12-02", "buy", "VOO", 2, 600.0, 0, 0, 38500, "USD buy, TWD cash out"],
        ["2026-12-20", "dividend", "006208.TW", "", "", "", "", 120, ""],
        ["2026-12-21", "sell", "006208.TW", 10, 200.0, 3, 2, "", ""],
    ]


def test_ledger_cash_and_positions(tmp_path: Path):
    led = el.load_ledger(_write(tmp_path, _rows()))
    cash = el.cash_balance(led, "2026-12-31")
    assert cash == pytest.approx(100000 - (79 * 190 + 21) - 38500 + 120 + (10 * 200 - 3 - 2))
    assert el.positions(led, "2026-12-31") == {"006208.TW": 69, "VOO": 2}
    assert el.positions(led, "2026-12-02") == {"006208.TW": 79, "VOO": 2}
    flows = el.external_flows(led)
    assert flows.sum() == pytest.approx(100000)


def test_ledger_rejects_unknown_type_and_overselling(tmp_path: Path):
    bad = _rows() + [["2026-12-22", "gift", "", "", "", "", "", 1, ""]]
    with pytest.raises(ValueError, match="type"):
        el.load_ledger(_write(tmp_path, bad))
    oversell = _rows() + [["2026-12-23", "sell", "VOO", 5, 600.0, 0, 0, 1000, ""]]
    with pytest.raises(ValueError, match="negative"):
        el.load_ledger(_write(tmp_path, oversell))


def test_ledger_rejects_negative_cash(tmp_path: Path):
    rows = [["2026-12-01", "deposit", "", "", "", "", "", 1000, ""],
            ["2026-12-02", "buy", "0050.TW", 100, 80.0, 1, 0, "", ""]]
    with pytest.raises(ValueError, match="cash"):
        el.load_ledger(_write(tmp_path, rows))


def test_init_template_writes_header_only(tmp_path: Path):
    p = tmp_path / "new" / "ledger.csv"
    el.init_ledger(p)
    assert pd.read_csv(p).columns.tolist() == el.COLUMNS
    with pytest.raises(FileExistsError):
        el.init_ledger(p)


def test_empty_ledger_gives_clear_error(tmp_path: Path):
    p = tmp_path / "ledger.csv"
    el.init_ledger(p)
    with pytest.raises(ValueError, match="empty"):
        el.load_ledger(p)
