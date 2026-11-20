"""Regression tests for the P0 live-money safety fixes in `broker_exec.py` / `exec_safety.py`.

Each test encodes a concrete failure scenario found in the red-team audit:
duplicate orders, dead drawdown breaker, over-committed cash, invisible Shioaji
odd-lot / OTC positions, and trading a scoring model that was never backtested.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

import broker_exec as bx
import exec_safety as es

TRADE_DATE = "2026-11-20"


class _FakeAdapter(bx.BaseBrokerAdapter):
    """Configurable offline broker used to drive `execute_order_plan`."""

    name = "alpaca"

    def __init__(self, equity=20000.0, cash=5000.0, positions=None, pending=None, submitted=None, fail_first=None):
        self.equity = equity
        self.cash = cash
        self.positions = dict(positions or {})
        self.pending = dict(pending or {})
        self.submitted = set(submitted or set())
        self.fail_first = fail_first
        self.calls = 0

    def get_account_snapshot(self):
        return {
            "broker": self.name,
            "equity_usd": self.equity,
            "cash_usd": self.cash,
            "available_cash_usd": self.cash,
            "positions_usd": dict(self.positions),
        }

    def get_order_state(self, trade_date):
        return {"pending_buy_usd": dict(self.pending), "submitted_keys": set(self.submitted)}

    def submit_limit_order(self, order):
        self.calls += 1
        if self.fail_first is not None and self.calls == 1:
            raise self.fail_first
        return {"id": f"ord_{self.calls}", "status": "accepted"}


def _buy(ticker, qty, px, market="US", trade_date=TRADE_DATE, broker="alpaca"):
    cid = bx.make_client_order_id(trade_date, broker, ticker, "buy", market=market)
    ccy = "TWD" if market == "TW" else "USD"
    fx = 1.0 / 32.0 if market == "TW" else 1.0
    return bx.OrderIntent(ticker, market, "buy", float(qty), px, px, ccy, round(qty * px * fx, 2), "test", cid)


def _run(plan, adapter, tmp_path, **kw):
    return bx.execute_order_plan(
        plan, adapter, bx.RiskLimits(), run_date=TRADE_DATE,
        ledger_path=tmp_path / "ledger.json", audit_dir=tmp_path, **kw,
    )


# ---------------------------------------------------------------- P0-3 duplicates

def test_client_order_id_is_independent_of_quantity():
    a = bx.make_client_order_id(TRADE_DATE, "alpaca", "AAPL", "buy", market="US")
    b = bx.make_client_order_id(TRADE_DATE, "alpaca", "aapl", "BUY", market="US")
    assert a == b
    assert a != bx.make_client_order_id("2026-11-21", "alpaca", "AAPL", "buy", market="US")


def test_same_day_rerun_with_different_qty_is_blocked(tmp_path):
    adapter = _FakeAdapter()
    _run([_buy("AAPL", 2, 200.0)], adapter, tmp_path)
    r2 = _run([_buy("AAPL", 3, 201.0)], adapter, tmp_path)
    assert adapter.calls == 1
    assert r2["summary"]["executed"] == 0
    assert "DUPLICATE_CLIENT_ORDER_ID" in r2["rejected"][0]["reason"]


def test_submit_timeout_is_recorded_unknown_and_blocks_retry(tmp_path):
    adapter = _FakeAdapter(fail_first=requests.Timeout("read timed out"))
    r1 = _run([_buy("AAPL", 2, 200.0)], adapter, tmp_path)
    assert r1["summary"]["errors"] == 1
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert list(ledger["orders"].values())[0]["status"] == "UNKNOWN_NEEDS_RECONCILE"

    r2 = _run([_buy("AAPL", 2, 200.0)], adapter, tmp_path)
    assert adapter.calls == 1
    assert r2["summary"]["executed"] == 0


def test_broker_side_submitted_key_blocks_order(tmp_path):
    order = _buy("AAPL", 2, 200.0)
    adapter = _FakeAdapter(submitted={order.client_order_id})
    r = _run([order], adapter, tmp_path)
    assert adapter.calls == 0
    assert "ALREADY_AT_BROKER" in r["rejected"][0]["reason"]


def test_open_buy_orders_count_toward_position_weight(tmp_path):
    # 12% already pending at broker + 6% new = 18% > 15% ceiling
    adapter = _FakeAdapter(equity=10000.0, cash=5000.0, pending={"AAPL": 1200.0})
    r = _run([_buy("AAPL", 3, 200.0)], adapter, tmp_path)
    assert adapter.calls == 0
    assert "EXCEEDS_MAX_POSITION_WEIGHT" in r["rejected"][0]["reason"]


def test_order_state_failure_fails_closed(tmp_path):
    class _NoState(_FakeAdapter):
        def get_order_state(self, trade_date):
            raise requests.ConnectionError("down")

    adapter = _NoState()
    r = _run([_buy("AAPL", 2, 200.0)], adapter, tmp_path)
    assert adapter.calls == 0
    assert "ORDER_STATE_UNKNOWN" in r["rejected"][0]["reason"]


def test_exchange_trade_date_uses_exchange_timezone():
    now = dt.datetime(2026, 11, 20, 16, 30, tzinfo=dt.timezone.utc)  # 00:30 Sat Taipei, 11:30 Fri NY
    assert es.exchange_trade_date("TW", now) == "2026-11-21"
    assert es.exchange_trade_date("US", now) == "2026-11-20"


# ---------------------------------------------------------------- P0-2 drawdown breaker

def test_validate_without_drawdown_rejects_buys():
    approved, rejected = bx.validate_order_plan(
        [_buy("AAPL", 2, 200.0)], bx.RiskLimits(),
        account_equity_usd=10000.0, current_positions_usd={}, run_date=TRADE_DATE,
    )
    assert approved == []
    assert "DRAWDOWN_UNKNOWN" in rejected[0]["reason"]


def test_drawdown_is_computed_from_persisted_high_water_mark(tmp_path):
    _run([], _FakeAdapter(equity=100000.0), tmp_path)  # first run records HWM = 100k
    adapter = _FakeAdapter(equity=75000.0)  # -25% <= -20% halt
    r = _run([_buy("AAPL", 2, 200.0)], adapter, tmp_path)
    assert adapter.calls == 0
    assert "DRAWDOWN_HALT" in r["rejected"][0]["reason"]
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["equity_hwm_usd"]["alpaca"] == 100000.0


def test_drawdown_helper_math():
    assert es.drawdown_from_hwm(80.0, 100.0) == (pytest.approx(-0.20), 100.0)
    assert es.drawdown_from_hwm(120.0, 100.0) == (0.0, 120.0)
    assert es.drawdown_from_hwm(50.0, None) == (0.0, 50.0)
    with pytest.raises(ValueError):
        es.drawdown_from_hwm(0.0, 100.0)


def test_zero_equity_rejects_buys_instead_of_skipping_weight_check():
    approved, rejected = bx.validate_order_plan(
        [_buy("AAPL", 2, 200.0)], bx.RiskLimits(),
        account_equity_usd=0.0, current_positions_usd={}, run_date=TRADE_DATE, account_drawdown_pct=0.0,
    )
    assert approved == []
    assert "EQUITY_UNKNOWN" in rejected[0]["reason"]


# ---------------------------------------------------------------- P0-4 cash / margin

def test_deployable_cash_never_exceeds_available_minus_pending():
    assert es.deployable_cash_usd(50000.0, 10000.0, 2000.0) == 8000.0
    assert es.deployable_cash_usd(1000.0, 10000.0, 0.0) == 1000.0
    assert es.deployable_cash_usd(1000.0, 500.0, 900.0) == 0.0
    assert es.deployable_cash_usd(1000.0, -300.0, 0.0) == 0.0  # margin debit


def test_orders_beyond_available_cash_are_rejected(tmp_path):
    adapter = _FakeAdapter(equity=100000.0, cash=500.0)
    r = _run([_buy("AAPL", 2, 200.0), _buy("MSFT", 2, 200.0)], adapter, tmp_path)
    assert adapter.calls == 1
    assert "INSUFFICIENT_CASH" in r["rejected"][0]["reason"]


# ---------------------------------------------------------------- P0-5 Shioaji units / symbols

class _MockShioaji:
    stock_account = "SIM_ACC"

    def __init__(self):
        self.placed = []
        self.Contracts = SimpleNamespace(Stocks={"8299": "C_8299", "2330": "C_2330", "2412": "C_2412"})

    def list_positions(self, acc, unit=None):
        assert unit is not None, "positions must be requested in share units"
        return [
            SimpleNamespace(code="8299", quantity=500, last_price=640.0),
            SimpleNamespace(code="2330", quantity=1000, last_price=1000.0),
        ]

    def account_balance(self):
        return SimpleNamespace(acc_balance=320000.0)

    def settlements(self, acc):
        return [SimpleNamespace(date="2026-11-23", amount=-64000.0), SimpleNamespace(date="2026-11-24", amount=10000.0)]

    def update_status(self, acc):
        return None

    def list_trades(self):
        return [
            SimpleNamespace(
                contract=SimpleNamespace(code="2330"),
                order=SimpleNamespace(action="Buy", price=1000.0, quantity=1, order_lot="Common", custom_field="ABC123"),
                status=SimpleNamespace(status="Status.PartFilled", deal_quantity=0),
            ),
            SimpleNamespace(
                contract=SimpleNamespace(code="2412"),
                order=SimpleNamespace(action="Buy", price=120.0, quantity=100, order_lot="IntradayOdd", custom_field="DEF456"),
                status=SimpleNamespace(status="Status.Filled", deal_quantity=100),
            ),
        ]

    def Order(self, **kwargs):
        return kwargs

    def place_order(self, contract, order):
        self.placed.append(order)
        return SimpleNamespace(status=SimpleNamespace(id="sj_1", status="Submitted"))


def test_shioaji_positions_use_share_units_and_normalized_codes():
    adapter = bx.ShioajiBrokerAdapter(mode="paper", api_client=_MockShioaji(), twd_per_usd=32.0)
    snap = adapter.get_account_snapshot()
    assert snap["positions_usd"]["8299"] == pytest.approx(500 * 640.0 / 32.0)  # OTC odd lot visible
    assert snap["positions_usd"]["2330"] == pytest.approx(1000 * 1000.0 / 32.0)
    # Unsettled T+2 buy payable is deducted; incoming receivable is not counted
    assert snap["available_cash_usd"] == pytest.approx((320000.0 - 64000.0) / 32.0)


def test_otc_position_is_matched_by_weight_guard():
    order = _buy("8299.TWO", 50, 640.0, market="TW", broker="shioaji")  # $1,000 -> 9.9% + 1.0% > 10%
    approved, rejected = bx.validate_order_plan(
        [order], bx.RiskLimits(max_single_position_weight=0.10),
        account_equity_usd=100000.0, current_positions_usd={"8299": 9900.0},
        run_date=TRADE_DATE, account_drawdown_pct=0.0,
    )
    assert approved == []
    assert "EXCEEDS_MAX_POSITION_WEIGHT" in rejected[0]["reason"]


def test_shioaji_order_state_reports_open_buys_and_custom_tags():
    adapter = bx.ShioajiBrokerAdapter(mode="paper", api_client=_MockShioaji(), twd_per_usd=32.0)
    state = adapter.get_order_state(TRADE_DATE)
    assert state["pending_buy_usd"] == {"2330": pytest.approx(1000 * 1000.0 / 32.0)}
    assert state["submitted_keys"] == {"ABC123", "DEF456"}


def test_shioaji_custom_field_is_unique_per_order():
    api = _MockShioaji()
    adapter = bx.ShioajiBrokerAdapter(mode="paper", api_client=api, twd_per_usd=32.0)
    o1 = _buy("2330.TW", 10, 1000.0, market="TW", broker="shioaji")
    o2 = _buy("2412.TW", 10, 120.0, market="TW", broker="shioaji")
    o1.tw_lot_type = o2.tw_lot_type = "IntradayOdd"
    adapter.submit_limit_order(o1)
    adapter.submit_limit_order(o2)
    tags = [p["custom_field"] for p in api.placed]
    assert len(set(tags)) == 2 and all(len(t) == 6 for t in tags)
    assert adapter.order_dedupe_key(o1) == tags[0]


def test_tw_plan_never_emits_odd_lot_order_of_1000_or_more():
    df = pd.DataFrame({"ticker": ["2204.TW"], "close": [47.1], "v6_score": [3.0]})
    plan = bx.build_buy_only_plan(
        df, market="TW", cash_native=70000.0, current_positions_usd={}, account_equity_usd=100000.0,
        broker="shioaji", run_date=TRADE_DATE, limits=bx.RiskLimits(max_single_order_usd=5000.0), top_n=1,
    )
    assert len(plan) == 1
    o = plan[0]
    assert (o.tw_lot_type == "Common" and o.qty % 1000 == 0) or (o.tw_lot_type == "IntradayOdd" and o.qty < 1000)


# ---------------------------------------------------------------- symbols / ledger integrity

def test_alpaca_symbol_uses_dot_class_share_format(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "PK_TEST")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SK_TEST")
    sent = {}

    class _Resp:
        def __init__(self, p):
            self._p = p
        def raise_for_status(self):
            return None
        def json(self):
            return self._p

    class _Sess:
        def post(self, url, headers=None, json=None, timeout=15):
            sent.update(json or {})
            return _Resp({"id": "x", "status": "accepted"})

    adapter = bx.AlpacaBrokerAdapter(mode="paper", session=_Sess())
    adapter.submit_limit_order(_buy("BRK-B", 1, 500.0))
    assert sent["symbol"] == "BRK.B"


def test_corrupt_ledger_raises_instead_of_silently_resetting(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ledger"):
        bx.IdempotencyLedger(p)


# ---------------------------------------------------------------- P0-1 train/serve parity

def test_signal_parity_guard_blocks_neutral_fundamentals_outside_dry_run():
    w = {"mom_qual": 0.3, "ind_mom": 0.1, "norm_val": 0.3, "quality": 0.2, "fund": 0.05, "low_risk": 0.05}
    with pytest.raises(RuntimeError, match="neutral"):
        es.assert_signal_parity("paper", "neutral_fundamentals", w, accept_skew=False)
    es.assert_signal_parity("dry_run", "neutral_fundamentals", w, accept_skew=False)
    es.assert_signal_parity("paper", "csv", w, accept_skew=False)
    es.assert_signal_parity("paper", "neutral_fundamentals", w, accept_skew=True)


# ---------------------------------------------------------------- code-review follow-ups

def test_client_order_id_normalizes_symbol_spelling():
    tw = [bx.make_client_order_id(TRADE_DATE, "shioaji", t, "buy", market="TW") for t in ("2330.TW", "2330", "2330.tw")]
    us = [bx.make_client_order_id(TRADE_DATE, "alpaca", t, "buy", market="US") for t in ("BRK-B", "BRK.B")]
    assert len(set(tw)) == 1 and len(set(us)) == 1


class _AlpacaOrdersSession:
    """Mock Alpaca session: `status=open` returns queued orders from earlier days."""

    def __init__(self, open_orders, today_orders):
        self.open_orders, self.today_orders, self.calls = open_orders, today_orders, []

    def get(self, url, headers=None, params=None, timeout=15):
        self.calls.append(dict(params or {}))
        payload = self.open_orders if (params or {}).get("status") == "open" else self.today_orders
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


def _alpaca(monkeypatch, session):
    monkeypatch.setenv("ALPACA_API_KEY", "PK_TEST")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SK_TEST")
    return bx.AlpacaBrokerAdapter(mode="paper", session=session)


def test_alpaca_order_state_includes_orders_queued_on_earlier_days(monkeypatch):
    queued = [{"id": "o1", "client_order_id": "sb_sat", "symbol": "AAPL", "side": "buy", "status": "accepted",
               "qty": "5", "filled_qty": "0", "limit_price": "200"}]
    adapter = _alpaca(monkeypatch, _AlpacaOrdersSession(open_orders=queued, today_orders=[]))
    state = adapter.get_order_state(TRADE_DATE)
    assert state["pending_buy_usd"] == {"AAPL": pytest.approx(1000.0)}
    assert "sb_sat" in state["submitted_keys"]


def test_alpaca_unknown_live_status_counts_as_open(monkeypatch):
    orders = [{"id": "o2", "client_order_id": "x", "symbol": "MSFT", "side": "buy", "status": "pending_cancel",
               "qty": "2", "filled_qty": "0", "limit_price": "100"}]
    adapter = _alpaca(monkeypatch, _AlpacaOrdersSession(open_orders=[], today_orders=orders))
    assert adapter.get_order_state(TRADE_DATE)["pending_buy_usd"] == {"MSFT": pytest.approx(200.0)}


def test_alpaca_notional_order_is_valued_and_unpriced_order_fails_closed(monkeypatch):
    notional = [{"id": "o3", "client_order_id": "y", "symbol": "NVDA", "side": "buy", "status": "new",
                 "qty": None, "notional": "750", "filled_qty": "0", "filled_avg_price": None, "limit_price": None}]
    adapter = _alpaca(monkeypatch, _AlpacaOrdersSession(open_orders=notional, today_orders=[]))
    assert adapter.get_order_state(TRADE_DATE)["pending_buy_usd"] == {"NVDA": pytest.approx(750.0)}

    market = [{"id": "o4", "client_order_id": "z", "symbol": "TSLA", "side": "buy", "status": "new",
               "qty": "3", "filled_qty": "0", "limit_price": None, "notional": None}]
    adapter = _alpaca(monkeypatch, _AlpacaOrdersSession(open_orders=market, today_orders=[]))
    with pytest.raises(ValueError, match="TSLA"):
        adapter.get_order_state(TRADE_DATE)


def test_paper_and_live_use_separate_ledger_scopes(monkeypatch):
    adapter = _alpaca(monkeypatch, _AlpacaOrdersSession([], []))
    assert adapter.ledger_scope == "alpaca:paper"
    assert _FakeAdapter().ledger_scope == "alpaca"


def test_shioaji_equity_deducts_unsettled_payables():
    snap = bx.ShioajiBrokerAdapter(mode="paper", api_client=_MockShioaji(), twd_per_usd=32.0).get_account_snapshot()
    mv = 500 * 640.0 + 1000 * 1000.0
    assert snap["equity_usd"] == pytest.approx((mv + 320000.0 - 64000.0) / 32.0)


def test_ledger_write_failure_stops_further_submissions(tmp_path, monkeypatch):
    adapter = _FakeAdapter()

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(bx.IdempotencyLedger, "record", _boom)
    r = _run([_buy("AAPL", 2, 200.0), _buy("MSFT", 2, 200.0)], adapter, tmp_path)
    assert adapter.calls == 1
    assert any("LEDGER_WRITE_FAILED" in e["error"] for e in r["errors"])
    assert Path(r["audit_path"]).exists()
