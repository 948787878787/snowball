"""Broker adapters for the Snowball execution engine: Dry-Run, Alpaca REST v2, SinoPac Shioaji.

Every adapter exposes the same contract used by `broker_exec.execute_order_plan`:
`get_account_snapshot`, `get_order_state` (open buys + already-submitted keys for the
trade date), `order_dedupe_key` and `submit_limit_order`.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Set
from zoneinfo import ZoneInfo

import requests

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import exec_safety as es  # noqa: E402
from order_model import (  # noqa: E402
    ALPACA_LIVE_URL,
    ALPACA_PAPER_URL,
    DEFAULT_TWD_PER_USD,
    OrderIntent,
    assert_mode_allowed,
)

# Deny-lists: any status not known to be terminal is treated as still able to fill (fail closed).
ALPACA_CLOSED_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "replaced"})
SHIOAJI_CLOSED_STATUSES = frozenset({"filled", "cancelled", "failed"})


class BaseBrokerAdapter:
    """Abstract broker interface for Dry-Run, Alpaca Paper/Live, and SinoPac Shioaji."""

    name: str = "base"

    @property
    def ledger_scope(self) -> str:
        """Ledger key for HWM / ids; mode-qualified where a mode exists so paper never pollutes live."""
        mode = getattr(self, "mode", "")
        return f"{self.name}:{mode}" if mode else self.name

    def get_account_snapshot(self) -> Dict[str, Any]:
        """Return equity_usd, cash_usd, available_cash_usd (settled, non-margin) and positions_usd."""
        raise NotImplementedError

    def get_order_state(self, trade_date: str) -> Dict[str, Any]:
        """Return today's broker-side state: `pending_buy_usd` {symbol: usd still working}
        and `submitted_keys` (dedupe keys of every order already at the broker today)."""
        raise NotImplementedError

    def order_dedupe_key(self, order: OrderIntent) -> str:
        """Key under which the broker echoes our order back (client_order_id by default)."""
        return order.client_order_id

    def submit_limit_order(self, order: OrderIntent) -> Dict[str, Any]:
        raise NotImplementedError


class DryRunAdapter(BaseBrokerAdapter):
    """Zero-network simulator for testing order generation and pre-trade risk guards."""

    name = "dry_run"

    def __init__(self, equity_usd: float = 20000.0, cash_usd: float = 1500.0, positions_usd: Optional[Dict[str, float]] = None):
        self.equity_usd = equity_usd
        self.cash_usd = cash_usd
        self.positions_usd = dict(positions_usd or {})

    def get_account_snapshot(self) -> Dict[str, Any]:
        return {
            "broker": "dry_run",
            "mode": "dry_run",
            "equity_usd": self.equity_usd,
            "cash_usd": self.cash_usd,
            "available_cash_usd": self.cash_usd,
            "positions_usd": dict(self.positions_usd),
        }

    def get_order_state(self, trade_date: str) -> Dict[str, Any]:
        return {"pending_buy_usd": {}, "submitted_keys": set()}

    def submit_limit_order(self, order: OrderIntent) -> Dict[str, Any]:
        return {
            "id": f"dry_{order.client_order_id}",
            "status": "dry_run_simulated",
            "ticker": order.ticker,
            "side": order.side,
            "qty": order.qty,
            "limit_price": order.limit_price,
            "client_order_id": order.client_order_id,
        }


class AlpacaBrokerAdapter(BaseBrokerAdapter):
    """Alpaca REST v2 Adapter (Paper Trading default: https://paper-api.alpaca.markets).

    Zero extra SDK dependency — uses standard `requests` with strict API key env vars.
    """

    name = "alpaca"

    def __init__(self, mode: str = "paper", session: Optional[requests.Session] = None):
        assert_mode_allowed(mode)
        self.mode = mode.lower()
        self.base_url = ALPACA_LIVE_URL if self.mode == "live" else ALPACA_PAPER_URL
        self.api_key = os.environ.get("ALPACA_API_KEY", "").strip()
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()
        if not self.api_key or not self.secret_key:
            raise EnvironmentError(
                "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY. "
                "Create a free Paper Trading key at https://app.alpaca.markets/ and export both env vars."
            )
        self.session = session or requests.Session()
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def get_account_snapshot(self) -> Dict[str, Any]:
        r_acc = self.session.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=15)
        r_acc.raise_for_status()
        acc = r_acc.json()
        r_pos = self.session.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=15)
        r_pos.raise_for_status()
        pos_map = {es.normalize_symbol(p["symbol"], "US"): float(p.get("market_value", 0.0)) for p in r_pos.json()}
        cash = float(acc.get("cash", 0.0))
        # Never spend margin: cap by settled cash and by non-marginable buying power when reported
        nmbp = acc.get("non_marginable_buying_power")
        available = min(cash, float(nmbp)) if nmbp is not None else cash
        return {
            "broker": "alpaca",
            "mode": self.mode,
            "equity_usd": float(acc.get("equity", 0.0)),
            "cash_usd": cash,
            "available_cash_usd": max(available, 0.0),
            "positions_usd": pos_map,
        }

    def _list_orders(self, params: Dict[str, Any]) -> list:
        r = self.session.get(f"{self.base_url}/v2/orders", headers=self.headers, params=params, timeout=15)
        r.raise_for_status()
        return list(r.json())

    @staticmethod
    def _open_buy_value_usd(o: Dict[str, Any]) -> float:
        """Remaining USD a working buy can still consume; raises when it cannot be priced (fail closed)."""
        if o.get("qty") in (None, "") and o.get("notional") not in (None, ""):
            spent = float(o.get("filled_qty") or 0.0) * float(o.get("filled_avg_price") or 0.0)
            return max(float(o["notional"]) - spent, 0.0)
        if o.get("limit_price") in (None, ""):
            raise ValueError(
                f"Open Alpaca buy for {o.get('symbol')} (id {o.get('id')}) has no limit price or notional; "
                "cannot bound its cost. Cancel or let it fill before running Snowball."
            )
        remaining = float(o.get("qty") or 0.0) - float(o.get("filled_qty") or 0.0)
        return max(remaining, 0.0) * float(o["limit_price"])

    def get_order_state(self, trade_date: str) -> Dict[str, Any]:
        """Union of ALL still-open orders (any submit date: weekend/after-hours queued day orders)
        and every order submitted since the start of this NY trade date."""
        day_start = dt.datetime.combine(dt.date.fromisoformat(trade_date), dt.time.min, tzinfo=ZoneInfo(es.EXCHANGE_TZ["US"]))
        open_orders = self._list_orders({"status": "open", "limit": 500, "direction": "desc"})
        today_orders = self._list_orders({
            "status": "all", "after": day_start.astimezone(dt.timezone.utc).isoformat(), "limit": 500, "direction": "desc",
        })
        unique = {str(o.get("id") or o.get("client_order_id")): o for o in [*open_orders, *today_orders]}
        pending: Dict[str, float] = {}
        keys: Set[str] = set()
        for o in unique.values():
            if o.get("client_order_id"):
                keys.add(str(o["client_order_id"]))
            if str(o.get("status", "")).lower() in ALPACA_CLOSED_STATUSES or str(o.get("side", "")).lower() != "buy":
                continue
            sym = es.normalize_symbol(o.get("symbol", ""), "US")
            pending[sym] = pending.get(sym, 0.0) + self._open_buy_value_usd(o)
        return {"pending_buy_usd": pending, "submitted_keys": keys}

    def submit_limit_order(self, order: OrderIntent) -> Dict[str, Any]:
        payload = {
            "symbol": es.normalize_symbol(order.ticker, "US"),
            "qty": str(order.qty),
            "side": order.side.lower(),
            "type": "limit",
            "time_in_force": "day",
            "limit_price": f"{order.limit_price:.2f}",
            "client_order_id": order.client_order_id,
        }
        r = self.session.post(f"{self.base_url}/v2/orders", headers=self.headers, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()


class ShioajiBrokerAdapter(BaseBrokerAdapter):
    """SinoPac Shioaji Adapter for Taiwan Equities (Simulation Mode default: `simulation=True`)."""

    name = "shioaji"

    def __init__(self, mode: str = "paper", api_client: Optional[Any] = None, twd_per_usd: float = DEFAULT_TWD_PER_USD):
        assert_mode_allowed(mode)
        self.mode = mode.lower()
        self.simulation = self.mode != "live"
        self.twd_per_usd = twd_per_usd
        self._api = api_client
        self._sj = None

        if self._api is None:
            api_key = os.environ.get("SHIOAJI_API_KEY", "").strip()
            secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "").strip()
            if not api_key or not secret_key:
                raise EnvironmentError(
                    "Missing SHIOAJI_API_KEY or SHIOAJI_SECRET_KEY. "
                    "Apply for SinoPac Shioaji API credentials and export both env vars."
                )
            try:
                import shioaji as sj
            except ImportError as exc:
                raise ImportError(
                    "The `shioaji` package is not installed. Install it via: pip install shioaji"
                ) from exc
            self._sj = sj
            self._api = sj.Shioaji(simulation=self.simulation)
            self._api.login(api_key=api_key, secret_key=secret_key)
            if not self.simulation:
                ca_path = os.environ.get("SHIOAJI_CA_PATH", "").strip()
                ca_pwd = os.environ.get("SHIOAJI_CA_PASSWORD", "").strip()
                if not ca_path or not ca_pwd:
                    raise EnvironmentError("Live Shioaji requires SHIOAJI_CA_PATH and SHIOAJI_CA_PASSWORD.")
                self._api.activate_ca(ca_path=ca_path, ca_passwd=ca_pwd)

    def _share_unit(self) -> Any:
        return self._sj.constant.Unit.Share if self._sj is not None else "Share"

    def get_account_snapshot(self) -> Dict[str, Any]:
        # Request SHARE units so odd-lot holdings (<1,000 shares) are visible to the weight guard
        positions = self._api.list_positions(self._api.stock_account, unit=self._share_unit()) or []
        pos_usd: Dict[str, float] = {}
        tot_mv_twd = 0.0
        for p in positions:
            code = es.normalize_symbol(str(getattr(p, "code", "")), "TW")
            mv_twd = float(getattr(p, "quantity", 0.0)) * float(getattr(p, "last_price", 0.0))
            tot_mv_twd += mv_twd
            if code:
                pos_usd[code] = pos_usd.get(code, 0.0) + mv_twd / self.twd_per_usd
        cash_twd = float(getattr(self._api.account_balance(), "acc_balance", 0.0))
        # T+1/T+2 payables for already-filled buys are not yet out of the bank balance; deduct them
        # (receivables from sells are ignored: conservative, avoids default settlement / 違約交割).
        payables_twd = sum(
            float(getattr(s, "amount", 0.0)) for s in (self._api.settlements(self._api.stock_account) or [])
            if float(getattr(s, "amount", 0.0)) < 0.0
        )
        return {
            "broker": "shioaji",
            "mode": self.mode,
            "simulation": self.simulation,
            # Payables are negative: equity excludes cash already owed for filled-but-unsettled buys
            "equity_usd": (tot_mv_twd + cash_twd + payables_twd) / self.twd_per_usd,
            "cash_usd": cash_twd / self.twd_per_usd,
            "available_cash_usd": max(cash_twd + payables_twd, 0.0) / self.twd_per_usd,
            "positions_usd": pos_usd,
        }

    def get_order_state(self, trade_date: str) -> Dict[str, Any]:
        self._api.update_status(self._api.stock_account)
        pending: Dict[str, float] = {}
        keys: Set[str] = set()
        for t in self._api.list_trades() or []:
            order = getattr(t, "order", None)
            status = getattr(t, "status", None)
            if order is None:
                continue
            tag = str(getattr(order, "custom_field", "") or "")
            if tag:
                keys.add(tag)
            state = str(getattr(status, "status", "")).rsplit(".", 1)[-1].lower()
            if state in SHIOAJI_CLOSED_STATUSES or "buy" not in str(getattr(order, "action", "")).lower():
                continue
            lot_mult = 1000.0 if str(getattr(order, "order_lot", "")).lower().endswith("common") else 1.0
            remaining = float(getattr(order, "quantity", 0)) - float(getattr(status, "deal_quantity", 0) or 0)
            code = es.normalize_symbol(str(getattr(getattr(t, "contract", None), "code", "")), "TW")
            usd = max(remaining, 0.0) * lot_mult * float(getattr(order, "price", 0.0)) / self.twd_per_usd
            pending[code] = pending.get(code, 0.0) + usd
        return {"pending_buy_usd": pending, "submitted_keys": keys}

    def order_dedupe_key(self, order: OrderIntent) -> str:
        return es.shioaji_custom_tag(order.client_order_id)

    def submit_limit_order(self, order: OrderIntent) -> Dict[str, Any]:
        bare_code = es.normalize_symbol(order.ticker, "TW")
        contract = self._api.Contracts.Stocks[bare_code]
        shares = int(order.qty)
        is_common = order.tw_lot_type == "Common"
        if is_common and shares % es.TW_BOARD_LOT != 0:
            raise ValueError(f"Board-lot order for {order.ticker} must be a multiple of 1,000 shares (got {shares}).")
        if not is_common and not 1 <= shares < es.TW_BOARD_LOT:
            raise ValueError(f"Intraday odd-lot order for {order.ticker} must be 1-999 shares (got {shares}).")
        order_qty = shares // es.TW_BOARD_LOT if is_common else shares
        order_lot = "Common" if is_common else "IntradayOdd"

        if self._sj is not None:
            action = self._sj.constant.Action.Buy if order.side == "buy" else self._sj.constant.Action.Sell
            price_type = self._sj.constant.StockPriceType.LMT
            ord_type = self._sj.constant.OrderType.ROD
            lot_enum = self._sj.constant.StockOrderLot.Common if is_common else self._sj.constant.StockOrderLot.IntradayOdd
        else:
            action = "Buy" if order.side == "buy" else "Sell"
            price_type = "LMT"
            ord_type = "ROD"
            lot_enum = order_lot

        sj_order = self._api.Order(
            price=order.limit_price,
            quantity=order_qty,
            action=action,
            price_type=price_type,
            order_type=ord_type,
            order_lot=lot_enum,
            custom_field=self.order_dedupe_key(order),
            account=self._api.stock_account,
        )
        trade = self._api.place_order(contract, sj_order)
        status_obj = getattr(trade, "status", None)
        return {
            "id": str(getattr(status_obj, "id", order.client_order_id)),
            "status": str(getattr(status_obj, "status", "submitted")),
            "simulation": self.simulation,
            "ticker": order.ticker,
            "order_lot": order_lot,
            "quantity": order_qty,
            "limit_price": order.limit_price,
            "client_order_id": order.client_order_id,
        }
