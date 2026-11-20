"""Offline unit tests for Snowball Safe Paper/Live Broker Execution Engine (`broker_exec.py`)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import broker_exec as bx


def test_live_mode_is_hard_blocked_without_explicit_unlock(monkeypatch):
    monkeypatch.delenv(bx.LIVE_UNLOCK_ENV, raising=False)
    bx.assert_mode_allowed("dry_run")
    bx.assert_mode_allowed("paper")
    with pytest.raises(PermissionError, match="SAFETY LOCK"):
        bx.assert_mode_allowed("live")

    monkeypatch.setenv(bx.LIVE_UNLOCK_ENV, "wrong_phrase")
    with pytest.raises(PermissionError):
        bx.assert_mode_allowed("live")

    monkeypatch.setenv(bx.LIVE_UNLOCK_ENV, bx.LIVE_UNLOCK_PHRASE)
    bx.assert_mode_allowed("live")


def test_tw_tick_rounding_and_limit_buffers():
    assert bx.tw_round_tick(47.12, side="buy") == 47.10
    assert bx.tw_round_tick(47.12, side="sell") == 47.15
    assert bx.tw_round_tick(1023.0, side="buy") == 1020.0
    assert bx.tw_round_tick(1023.0, side="sell") == 1025.0
    assert bx.compute_limit_price(100.0, side="buy", market="US", buffer_pct=0.005) == 100.50
    assert bx.compute_limit_price(100.0, side="sell", market="US", buffer_pct=0.005) == 99.50


def test_pre_trade_risk_guards_reject_oversized_overweight_and_killswitch(tmp_path: Path):
    limits = bx.RiskLimits(
        max_single_order_usd=1000.0,
        max_daily_notional_usd=1500.0,
        max_single_position_weight=0.15,
        max_drawdown_halt_pct=-0.20,
    )
    orders = [
        bx.OrderIntent("AAPL", "US", "buy", 4, 200.0, 201.0, "USD", 804.0, "ok", "cid_1"),
        bx.OrderIntent("MSFT", "US", "buy", 10, 200.0, 201.0, "USD", 2010.0, "too big", "cid_2"),
        bx.OrderIntent("NVDA", "US", "buy", 3, 200.0, 201.0, "USD", 603.0, "overweight", "cid_3"),
        bx.OrderIntent("GOOGL", "US", "buy", 4, 200.0, 201.0, "USD", 804.0, "daily cap", "cid_4"),
    ]
    approved, rejected = bx.validate_order_plan(
        orders,
        limits,
        account_equity_usd=10000.0,
        current_positions_usd={"NVDA": 1200.0},  # 12% + 6% = 18% > 15% ceiling
        run_date="2026-11-20",
        account_drawdown_pct=0.0,  # drawdown must be explicit; None fails closed
    )
    assert [o.ticker for o in approved] == ["AAPL"]
    reasons = {r["order"]["ticker"]: r["reason"] for r in rejected}
    assert "EXCEEDS_MAX_SINGLE_ORDER" in reasons["MSFT"]
    assert "EXCEEDS_MAX_POSITION_WEIGHT" in reasons["NVDA"]
    assert "EXCEEDS_DAILY_NOTIONAL" in reasons["GOOGL"]

    # Kill switch halts 100% of orders
    ks = tmp_path / ".snowball_kill_switch"
    ks.write_text("STOP", encoding="utf-8")
    app_ks, rej_ks = bx.validate_order_plan(
        orders[:1], limits, account_equity_usd=10000.0, current_positions_usd={}, run_date="2026-11-20", kill_switch_path=ks
    )
    assert app_ks == []
    assert "KILL_SWITCH_ACTIVE" in rej_ks[0]["reason"]

    # Drawdown halt blocks buys
    app_dd, rej_dd = bx.validate_order_plan(
        orders[:1], limits, account_equity_usd=10000.0, current_positions_usd={}, run_date="2026-11-20", account_drawdown_pct=-0.22
    )
    assert app_dd == []
    assert "DRAWDOWN_HALT" in rej_dd[0]["reason"]


def test_idempotency_ledger_blocks_duplicate_submission_on_second_run(tmp_path: Path):
    class _FakePaperAdapter(bx.BaseBrokerAdapter):
        name = "alpaca"
        def __init__(self):
            self.calls = 0
        def get_account_snapshot(self):
            return {"broker": "alpaca", "equity_usd": 20000.0, "cash_usd": 2000.0, "available_cash_usd": 2000.0, "positions_usd": {}}
        def get_order_state(self, trade_date):
            return {"pending_buy_usd": {}, "submitted_keys": set()}
        def submit_limit_order(self, order):
            self.calls += 1
            return {"id": f"ord_{self.calls}", "status": "accepted"}

    adapter = _FakePaperAdapter()
    limits = bx.RiskLimits()
    cid = bx.make_client_order_id("2026-11-20", "alpaca", "BRK-B", "buy", market="US")
    plan = [bx.OrderIntent("BRK-B", "US", "buy", 2.0, 500.0, 502.5, "USD", 1005.0, "test", cid)]
    ledger_path = tmp_path / "ledger.json"

    r1 = bx.execute_order_plan(plan, adapter, limits, run_date="2026-11-20", ledger_path=ledger_path, audit_dir=tmp_path)
    assert r1["summary"]["executed"] == 1
    assert adapter.calls == 1

    # Second identical run on same day MUST skip duplicate
    r2 = bx.execute_order_plan(plan, adapter, limits, run_date="2026-11-20", ledger_path=ledger_path, audit_dir=tmp_path)
    assert r2["summary"]["executed"] == 0
    assert r2["summary"]["rejected"] == 1
    assert "DUPLICATE_CLIENT_ORDER_ID" in r2["rejected"][0]["reason"]
    assert adapter.calls == 1


def test_alpaca_paper_adapter_uses_paper_endpoint_and_limit_payload(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "PK_TEST_123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SEC_TEST_456")

    recorded = {}

    class _MockResp:
        def __init__(self, payload):
            self._p = payload
        def raise_for_status(self):
            return None
        def json(self):
            return self._p

    class _MockSession:
        def get(self, url, headers=None, timeout=15):
            if url.endswith("/v2/account"):
                return _MockResp({"equity": "25000", "cash": "3000"})
            return _MockResp([{"symbol": "AAPL", "market_value": "2000"}])
        def post(self, url, headers=None, json=None, timeout=15):
            recorded["url"] = url
            recorded["headers"] = headers
            recorded["json"] = json
            return _MockResp({"id": "alp_999", "status": "accepted"})

    adapter = bx.AlpacaBrokerAdapter(mode="paper", session=_MockSession())
    snap = adapter.get_account_snapshot()
    assert snap["equity_usd"] == 25000.0
    assert snap["positions_usd"]["AAPL"] == 2000.0

    order = bx.OrderIntent("BRK-B", "US", "buy", 1.0, 574.62, 577.49, "USD", 577.49, "top", "sb_test_1")
    resp = adapter.submit_limit_order(order)
    assert resp["id"] == "alp_999"
    assert recorded["url"] == f"{bx.ALPACA_PAPER_URL}/v2/orders"
    assert recorded["headers"]["APCA-API-KEY-ID"] == "PK_TEST_123"
    assert recorded["json"]["type"] == "limit"
    assert recorded["json"]["time_in_force"] == "day"
    assert recorded["json"]["limit_price"] == "577.49"


def test_shioaji_simulation_adapter_handles_odd_lot_and_whole_lot():
    placed = []

    class _MockShioaji:
        stock_account = "SIM_ACC"
        Contracts = SimpleNamespace(Stocks={"2330": "CONTRACT_2330", "2204": "CONTRACT_2204"})
        def list_positions(self, acc, unit=None):
            return [SimpleNamespace(code="2330", quantity=1000, last_price=1000.0)]  # share units
        def account_balance(self):
            return SimpleNamespace(acc_balance=320000.0)
        def settlements(self, acc):
            return []
        def Order(self, **kwargs):
            return kwargs
        def place_order(self, contract, order):
            placed.append((contract, order))
            return SimpleNamespace(status=SimpleNamespace(id="sj_1", status="Submitted"))

    adapter = bx.ShioajiBrokerAdapter(mode="paper", api_client=_MockShioaji(), twd_per_usd=32.0)
    assert adapter.simulation is True
    snap = adapter.get_account_snapshot()
    assert snap["positions_usd"]["2330"] == pytest.approx(1000000.0 / 32.0)

    # Odd lot (300 shares -> IntradayOdd, qty=300)
    o_odd = bx.OrderIntent("2204.TW", "TW", "buy", 300.0, 47.1, 47.3, "TWD", 443.4, "odd", bx.make_client_order_id("2026-11-20", "shioaji", "2204.TW", "buy", market="TW"), "IntradayOdd")
    r_odd = adapter.submit_limit_order(o_odd)
    assert r_odd["order_lot"] == "IntradayOdd"
    assert r_odd["quantity"] == 300

    # Whole lot (2,000 shares -> Common, qty=2 lots)
    o_lot = bx.OrderIntent("2204.TW", "TW", "buy", 2000.0, 47.1, 47.3, "TWD", 2956.2, "lot", bx.make_client_order_id("2026-11-21", "shioaji", "2204.TW", "buy", market="TW"), "Common")
    r_lot = adapter.submit_limit_order(o_lot)
    assert r_lot["order_lot"] == "Common"
    assert r_lot["quantity"] == 2


def test_buy_only_planner_generates_valid_us_and_tw_plans():
    df_us = pd.DataFrame({
        "ticker": ["BRK-B", "KDP", "NTNX", "QGEN", "AAPL", "MSFT", "GOOGL", "TSLA"],
        "close": [574.6, 35.0, 57.4, 49.6, 352.5, 508.3, 282.4, 383.7],
        "v6_score": [3.0, 2.8, 2.5, 2.2, 1.2, -0.2, -0.3, -2.4],
    })
    limits = bx.RiskLimits(max_single_order_usd=1000.0, max_daily_notional_usd=3000.0)
    plan = bx.build_buy_only_plan(
        df_us, market="US", cash_native=1500.0, current_positions_usd={}, account_equity_usd=20000.0,
        broker="alpaca", run_date="2026-11-20", limits=limits, top_n=3
    )
    assert len(plan) >= 2
    assert all(o.side == "buy" and o.limit_price > o.ref_price for o in plan)
    assert sum(o.notional_usd for o in plan) <= 1550.0
