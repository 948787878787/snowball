#!/usr/bin/env python3
"""Safe Paper-Trading & Broker API Execution Engine (US Alpaca + TW SinoPac Shioaji).

Built with institutional pre-trade safety guards so research signals from Snowball (`global_ranker.py`
/ `v6_learned_weights.json`) can be tested safely in Paper Trading before ever touching real money:

4 Mandatory Safety Guards:
1. Dry-Run & Paper-Only Default:
   - Default `--mode dry_run` (zero network order calls; prints validated limit order plan & audit JSON).
   - `--mode paper` routes strictly to Alpaca Paper (`https://paper-api.alpaca.markets`) or
     SinoPac Shioaji (`simulation=True`).
   - `--mode live` is HARD-BLOCKED unless env `SNOWBALL_ALLOW_LIVE_ORDERS=YES_I_UNDERSTAND` is set.
2. Pre-Trade Risk Guard (`RiskLimits`):
   - Enforces `max_single_order_usd`, `max_daily_notional_usd`, `max_single_position_weight`,
     `max_orders_per_run`, and Kill Switch (`.snowball_kill_switch` file or drawdown halt).
   - Strictly uses **LIMIT orders** with configurable slippage buffer (`+0.5%` buy / `-0.5%` sell)
     and exact TWSE tick-size rounding for Taiwan equities — never submits blind market orders.
3. Idempotency / Anti-Duplicate Order Lock (`IdempotencyLedger`):
   - Assigns deterministic `client_order_id` (`sha256(date|broker|ticker|side|qty)`) persisted
     atomically on disk (`execution_ledger.json`). Re-running twice on the same day skips duplicates.
4. Zero Hardcoded Credentials:
   - Reads keys strictly from environment variables (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
     `SHIOAJI_API_KEY`, `SHIOAJI_SECRET_KEY`).

Usage:
    # 1. Zero-risk local dry-run from Snowball offline/live signals
    python broker_exec.py --broker alpaca --mode dry_run --cash 1500 --offline-demo

    # 2. Submit validated limit orders to Alpaca FREE Paper Trading API
    export ALPACA_API_KEY="PK..." ALPACA_SECRET_KEY="..."
    python broker_exec.py --broker alpaca --mode paper --cash 1500 --learned-weights ./v6_learned_weights.json

    # 3. Submit validated Taiwan odd-lot/whole-lot limit orders to SinoPac Shioaji Simulation
    export SHIOAJI_API_KEY="..." SHIOAJI_SECRET_KEY="..."
    python broker_exec.py --broker shioaji --mode paper --cash 50000 --learned-weights ./v6_learned_weights.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
LIVE_UNLOCK_ENV = "SNOWBALL_ALLOW_LIVE_ORDERS"
LIVE_UNLOCK_PHRASE = "YES_I_UNDERSTAND"
DEFAULT_TWD_PER_USD = 32.0


@dataclass(frozen=True)
class RiskLimits:
    """Institutional pre-trade risk limits checked before ANY order leaves the machine."""

    max_single_order_usd: float = 1500.0
    max_daily_notional_usd: float = 5000.0
    max_single_position_weight: float = 0.15
    max_orders_per_run: int = 10
    limit_buffer_pct: float = 0.005  # +0.5% for buy limit, -0.5% for sell limit
    max_drawdown_halt_pct: float = -0.20  # Halt all buys if account drawdown <= -20%
    min_order_notional_usd: float = 25.0
    twd_per_usd: float = DEFAULT_TWD_PER_USD


@dataclass
class OrderIntent:
    """Normalized order intent before and after pre-trade validation."""

    ticker: str
    market: str  # "US" or "TW"
    side: str  # "buy" or "sell"
    qty: float
    ref_price: float
    limit_price: float
    currency: str  # "USD" or "TWD"
    notional_usd: float
    reason: str
    client_order_id: str = ""
    tw_lot_type: str = ""  # "Common" (1,000-share board lot) or "IntradayOdd" (<1,000 shares)


def tw_round_tick(price: float, side: str = "buy") -> float:
    """Round price to official TWSE/TPEx price tick increments."""
    p = max(float(price), 0.01)
    if p < 10.0:
        tick = 0.01
    elif p < 50.0:
        tick = 0.05
    elif p < 100.0:
        tick = 0.10
    elif p < 500.0:
        tick = 0.50
    elif p < 1000.0:
        tick = 1.00
    else:
        tick = 5.00
    steps = np.floor(p / tick) if side == "buy" else np.ceil(p / tick)
    return round(float(steps * tick), 2)


def compute_limit_price(ref_price: float, side: str, market: str, buffer_pct: float = 0.005) -> float:
    """Compute protective limit price (+buffer for buys, -buffer for sells)."""
    raw = ref_price * (1.0 + buffer_pct) if side == "buy" else ref_price * (1.0 - buffer_pct)
    if market.upper() == "TW":
        return tw_round_tick(raw, side=side)
    return round(max(float(raw), 0.01), 2)


def make_client_order_id(run_date: str, broker: str, ticker: str, side: str, qty: float) -> str:
    """Deterministic idempotency key preventing duplicate orders on repeated script runs."""
    raw = f"{run_date}|{broker.lower()}|{ticker.upper()}|{side.lower()}|{qty:.4f}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"sb_{run_date.replace('-', '')}_{digest}"


def assert_mode_allowed(mode: str) -> None:
    """Hard lock preventing accidental live-money execution."""
    m = mode.lower().strip()
    if m not in {"dry_run", "paper", "live"}:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: dry_run, paper, live.")
    if m == "live":
        unlocked = os.environ.get(LIVE_UNLOCK_ENV, "").strip()
        if unlocked != LIVE_UNLOCK_PHRASE:
            raise PermissionError(
                f"LIVE ORDER BLOCKED BY SAFETY LOCK! To enable real-money execution, explicitly set "
                f"{LIVE_UNLOCK_ENV}={LIVE_UNLOCK_PHRASE} in your environment."
            )


class IdempotencyLedger:
    """Atomic JSON ledger tracking submitted client_order_ids and daily notional."""

    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Any] = {"orders": {}, "daily_notional_usd": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data.update(loaded)
            except Exception:
                pass

    def has_order(self, client_order_id: str) -> bool:
        return client_order_id in self._data.get("orders", {})

    def daily_notional(self, run_date: str) -> float:
        return float(self._data.get("daily_notional_usd", {}).get(run_date, 0.0))

    def record(self, order: OrderIntent, run_date: str, broker_resp: dict) -> None:
        self._data.setdefault("orders", {})[order.client_order_id] = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_date": run_date,
            "ticker": order.ticker,
            "side": order.side,
            "qty": order.qty,
            "limit_price": order.limit_price,
            "notional_usd": round(order.notional_usd, 2),
            "status": broker_resp.get("status", "submitted"),
            "broker_order_id": broker_resp.get("id", ""),
        }
        prev = self.daily_notional(run_date)
        self._data.setdefault("daily_notional_usd", {})[run_date] = round(prev + order.notional_usd, 2)
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)


def validate_order_plan(
    orders: Sequence[OrderIntent],
    limits: RiskLimits,
    *,
    account_equity_usd: float,
    current_positions_usd: Dict[str, float],
    run_date: str,
    ledger: Optional[IdempotencyLedger] = None,
    kill_switch_path: Optional[Path] = None,
    account_drawdown_pct: float = 0.0,
) -> Tuple[List[OrderIntent], List[Dict[str, Any]]]:
    """Validate every proposed order against all pre-trade risk guards.

    Returns `(approved_orders, rejected_records)`.
    """
    approved: List[OrderIntent] = []
    rejected: List[Dict[str, Any]] = []

    if kill_switch_path is not None and kill_switch_path.exists():
        for o in orders:
            rejected.append({"order": asdict(o), "reason": f"KILL_SWITCH_ACTIVE ({kill_switch_path})"})
        return approved, rejected

    if account_drawdown_pct <= limits.max_drawdown_halt_pct:
        for o in orders:
            if o.side == "buy":
                rejected.append({
                    "order": asdict(o),
                    "reason": f"DRAWDOWN_HALT ({account_drawdown_pct:.1%} <= {limits.max_drawdown_halt_pct:.1%})",
                })
            else:
                approved.append(o)
        return approved, rejected

    already_spent_today = ledger.daily_notional(run_date) if ledger is not None else 0.0
    running_notional = already_spent_today

    for o in orders:
        if len(approved) >= limits.max_orders_per_run:
            rejected.append({"order": asdict(o), "reason": f"MAX_ORDERS_PER_RUN ({limits.max_orders_per_run})"})
            continue
        if o.qty <= 0 or o.limit_price <= 0:
            rejected.append({"order": asdict(o), "reason": "INVALID_QTY_OR_PRICE"})
            continue
        if o.notional_usd < limits.min_order_notional_usd:
            rejected.append({"order": asdict(o), "reason": f"BELOW_MIN_NOTIONAL (${o.notional_usd:.2f} < ${limits.min_order_notional_usd:.2f})"})
            continue
        if o.notional_usd > limits.max_single_order_usd:
            rejected.append({"order": asdict(o), "reason": f"EXCEEDS_MAX_SINGLE_ORDER (${o.notional_usd:.2f} > ${limits.max_single_order_usd:.2f})"})
            continue
        if running_notional + o.notional_usd > limits.max_daily_notional_usd:
            rejected.append({
                "order": asdict(o),
                "reason": f"EXCEEDS_DAILY_NOTIONAL (${running_notional + o.notional_usd:.2f} > ${limits.max_daily_notional_usd:.2f})",
            })
            continue
        if o.side == "buy" and account_equity_usd > 0:
            cur_val = float(current_positions_usd.get(o.ticker, 0.0))
            post_weight = (cur_val + o.notional_usd) / account_equity_usd
            if post_weight > limits.max_single_position_weight + 1e-9:
                rejected.append({
                    "order": asdict(o),
                    "reason": f"EXCEEDS_MAX_POSITION_WEIGHT ({post_weight:.1%} > {limits.max_single_position_weight:.1%})",
                })
                continue
        if ledger is not None and o.client_order_id and ledger.has_order(o.client_order_id):
            rejected.append({"order": asdict(o), "reason": f"DUPLICATE_CLIENT_ORDER_ID ({o.client_order_id})"})
            continue

        running_notional += o.notional_usd
        approved.append(o)

    return approved, rejected


def build_buy_only_plan(
    ranked_df: pd.DataFrame,
    *,
    market: str,
    cash_native: float,
    current_positions_usd: Dict[str, float],
    account_equity_usd: float,
    broker: str,
    run_date: str,
    limits: RiskLimits,
    top_n: int = 5,
    target_weight: float = 0.08,
    allow_fractional_us: bool = False,
) -> List[OrderIntent]:
    """Convert Snowball ranked DataFrame into Buy-Only cash-flow limit orders.

    Uses Snowball Priority = `score + (w_target - w_now) / w_target` over Top-20% candidates,
    splitting available cash across top eligible names while respecting position weight ceilings.
    """
    if ranked_df.empty or cash_native <= 0:
        return []

    mkt = market.upper()
    currency = "TWD" if mkt == "TW" else "USD"
    fx_to_usd = (1.0 / limits.twd_per_usd) if currency == "TWD" else 1.0
    score_col = "v6_score" if "v6_score" in ranked_df.columns else ("v5_score" if "v5_score" in ranked_df.columns else ranked_df.columns[-1])

    df = ranked_df.dropna(subset=["ticker", "close", score_col]).copy()
    df = df[df["close"] > 0].sort_values(score_col, ascending=False).reset_index(drop=True)
    # Restrict buys strictly to Top-20% tier (at least top_n candidates)
    top_cut = max(top_n, int(np.ceil(len(df) * 0.20)))
    cands = df.iloc[:top_cut].copy()

    eq = max(account_equity_usd, cash_native * fx_to_usd)
    priorities = []
    for _, row in cands.iterrows():
        sym = str(row["ticker"])
        cur_w = float(current_positions_usd.get(sym, 0.0)) / eq
        if cur_w >= limits.max_single_position_weight - 0.005:
            continue
        prio = float(row[score_col]) + (target_weight - cur_w) / max(target_weight, 0.01)
        priorities.append((prio, sym, float(row["close"]), float(row[score_col]), cur_w))

    priorities.sort(key=lambda x: -x[0])
    chosen = priorities[:top_n]
    if not chosen:
        return []

    per_slot_native = cash_native / len(chosen)
    orders: List[OrderIntent] = []
    for prio, sym, ref_px, sc, cur_w in chosen:
        # Cap slot allocation by remaining single-position weight room and single-order limit
        max_add_usd = max(0.0, (limits.max_single_position_weight - cur_w) * eq)
        slot_usd = min(per_slot_native * fx_to_usd, max_add_usd, limits.max_single_order_usd)
        slot_native = slot_usd / fx_to_usd

        lim_px = compute_limit_price(ref_px, side="buy", market=mkt, buffer_pct=limits.limit_buffer_pct)
        if mkt == "TW":
            raw_sh = int(np.floor(slot_native / lim_px))
            if raw_sh <= 0:
                continue
            qty = float(raw_sh)
            lot_type = "Common" if (raw_sh >= 1000 and raw_sh % 1000 == 0) else "IntradayOdd"
        else:
            qty = round(slot_native / lim_px, 4) if allow_fractional_us else float(int(np.floor(slot_native / lim_px)))
            if qty <= 0:
                continue
            lot_type = ""

        notional_usd = qty * lim_px * fx_to_usd
        cid = make_client_order_id(run_date, broker, sym, "buy", qty)
        orders.append(
            OrderIntent(
                ticker=sym,
                market=mkt,
                side="buy",
                qty=qty,
                ref_price=round(ref_px, 4),
                limit_price=lim_px,
                currency=currency,
                notional_usd=round(notional_usd, 2),
                reason=f"Snowball Buy-Only Top-20% (score={sc:+.2f}, prio={prio:+.2f}, w={cur_w:.1%})",
                client_order_id=cid,
                tw_lot_type=lot_type,
            )
        )
    return orders


class BaseBrokerAdapter:
    """Abstract broker interface for Dry-Run, Alpaca Paper/Live, and SinoPac Shioaji."""

    name: str = "base"

    def get_account_snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError

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
            "positions_usd": dict(self.positions_usd),
        }

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
        pos_map = {str(p["symbol"]).upper(): float(p.get("market_value", 0.0)) for p in r_pos.json()}
        return {
            "broker": "alpaca",
            "mode": self.mode,
            "equity_usd": float(acc.get("equity", 0.0)),
            "cash_usd": float(acc.get("cash", 0.0)),
            "positions_usd": pos_map,
        }

    def submit_limit_order(self, order: OrderIntent) -> Dict[str, Any]:
        payload = {
            "symbol": order.ticker.upper(),
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

    def get_account_snapshot(self) -> Dict[str, Any]:
        positions = self._api.list_positions(self._api.stock_account) or []
        pos_usd: Dict[str, float] = {}
        tot_mv_twd = 0.0
        for p in positions:
            code = str(getattr(p, "code", "")).strip()
            qty = float(getattr(p, "quantity", 0.0))
            last_px = float(getattr(p, "last_price", 0.0))
            # Shioaji board-lot positions report quantity in lots (1 lot = 1,000 shares)
            mv_twd = qty * 1000.0 * last_px
            tot_mv_twd += mv_twd
            if code:
                pos_usd[f"{code}.TW"] = mv_twd / self.twd_per_usd
        bal = self._api.account_balance()
        cash_twd = float(getattr(bal, "acc_balance", 0.0))
        return {
            "broker": "shioaji",
            "mode": self.mode,
            "simulation": self.simulation,
            "equity_usd": (tot_mv_twd + cash_twd) / self.twd_per_usd,
            "cash_usd": cash_twd / self.twd_per_usd,
            "positions_usd": pos_usd,
        }

    def submit_limit_order(self, order: OrderIntent) -> Dict[str, Any]:
        bare_code = order.ticker.upper().replace(".TWO", "").replace(".TW", "")
        contract = self._api.Contracts.Stocks[bare_code]
        is_common = order.tw_lot_type == "Common" or (order.qty >= 1000 and int(order.qty) % 1000 == 0)
        order_qty = int(order.qty // 1000) if is_common else int(order.qty)
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
            custom_field=order.client_order_id[:6],
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


def execute_order_plan(
    orders: Sequence[OrderIntent],
    adapter: BaseBrokerAdapter,
    limits: RiskLimits,
    *,
    run_date: str,
    ledger_path: Path,
    audit_dir: Path,
    kill_switch_path: Optional[Path] = None,
    account_drawdown_pct: float = 0.0,
) -> Dict[str, Any]:
    """Validate, execute (or dry-run), record idempotency ledger, and write JSON audit log."""
    snap = adapter.get_account_snapshot()
    ledger = IdempotencyLedger(ledger_path)

    approved, rejected = validate_order_plan(
        orders,
        limits,
        account_equity_usd=float(snap.get("equity_usd", 0.0)),
        current_positions_usd=dict(snap.get("positions_usd", {})),
        run_date=run_date,
        ledger=ledger,
        kill_switch_path=kill_switch_path,
        account_drawdown_pct=account_drawdown_pct,
    )

    executed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for o in approved:
        try:
            resp = adapter.submit_limit_order(o)
            # Only record in real idempotency ledger when actually submitted (not dry_run)
            if adapter.name != "dry_run":
                ledger.record(o, run_date, resp)
            executed.append({"order": asdict(o), "response": resp})
        except Exception as exc:
            errors.append({"order": asdict(o), "error": str(exc)})

    report = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_date": run_date,
        "broker": adapter.name,
        "account_snapshot": snap,
        "risk_limits": asdict(limits),
        "summary": {
            "proposed": len(orders),
            "approved": len(approved),
            "executed": len(executed),
            "rejected": len(rejected),
            "errors": len(errors),
        },
        "executed": executed,
        "rejected": rejected,
        "errors": errors,
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    audit_path = audit_dir / f"order_audit_{adapter.name}_{ts_tag}.json"
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["audit_path"] = str(audit_path)
    return report


def _load_ranked_signals(
    market: str,
    learned_weights_path: Optional[Path],
    offline_demo: bool,
    signals_csv: Optional[Path],
) -> pd.DataFrame:
    if signals_csv is not None:
        return pd.read_csv(signals_csv)

    import global_ranker as gr
    from macro_micro import compute_macro_regime_panel, download_macro_series, make_offline_macro_data

    groups, tickers, benchmark, _, _, _, convert_usd = gr.resolve_universe_selection(market=market)
    if offline_demo:
        close_d, open_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
        macro_d = make_offline_macro_data(close_d.index)
    else:
        close_d, open_d, bench_d = gr.download_global_prices(tickers, benchmark, period="2y")
        if convert_usd:
            close_d, open_d = gr.convert_prices_to_usd(close_d, open_d, period="2y")
        macro_d = download_macro_series(period="2y")
        fund_df = gr.make_neutral_fundamentals(list(close_d.columns))

    pf = gr.compute_global_price_factors(close_d, bench_d, groups, open_d=open_d)
    macro_panel = compute_macro_regime_panel(macro_d, pf["close_w"].index)
    latest_macro = macro_panel.iloc[-1].to_dict() if not macro_panel.empty else None
    learned_w = gr.load_learned_weights(learned_weights_path) if learned_weights_path else None
    return gr.score_latest_global(pf, fund_df, groups, model="v6", macro_regime=latest_macro, learned_weights=learned_w)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball Safe Paper/Live Broker Execution Engine (Alpaca + Shioaji)")
    parser.add_argument("--broker", default="alpaca", choices=["alpaca", "shioaji"], help="Target broker adapter")
    parser.add_argument("--mode", default="dry_run", choices=["dry_run", "paper", "live"], help="Execution mode (default: dry_run)")
    parser.add_argument("--cash", type=float, default=1500.0, help="Cash budget to deploy in broker native currency (USD for Alpaca, TWD for Shioaji)")
    parser.add_argument("--top-n", type=int, default=4, help="Max number of Top-20% candidates to allocate cash across")
    parser.add_argument("--max-order-usd", type=float, default=1500.0, help="Hard cap per single order in USD")
    parser.add_argument("--max-daily-usd", type=float, default=5000.0, help="Hard daily total notional cap in USD")
    parser.add_argument("--max-weight", type=float, default=0.15, help="Hard single-stock portfolio weight ceiling")
    parser.add_argument("--signals-csv", type=Path, default=None, help="Optional pre-computed signals CSV (columns: ticker, close, v6_score)")
    parser.add_argument("--learned-weights", type=Path, default=None, help="Optional v6_learned_weights.json from Kaggle/Walk-Forward")
    parser.add_argument("--ledger", type=Path, default=Path("execution_ledger.json"), help="Idempotency ledger JSON path")
    parser.add_argument("--audit-dir", type=Path, default=Path("results"), help="Directory for order audit JSON logs")
    parser.add_argument("--kill-switch", type=Path, default=Path(".snowball_kill_switch"), help="Kill-switch file path (if present, halts all orders)")
    parser.add_argument("--offline-demo", action="store_true", help="Generate synthetic signals offline for end-to-end safety testing")
    args = parser.parse_args()

    assert_mode_allowed(args.mode)
    market = "TW" if args.broker == "shioaji" else "US"
    limits = RiskLimits(
        max_single_order_usd=args.max_order_usd,
        max_daily_notional_usd=args.max_daily_usd,
        max_single_position_weight=args.max_weight,
    )

    if args.mode == "dry_run":
        cash_usd = args.cash / limits.twd_per_usd if market == "TW" else args.cash
        adapter: BaseBrokerAdapter = DryRunAdapter(equity_usd=max(20000.0, cash_usd * 10.0), cash_usd=cash_usd)
    elif args.broker == "alpaca":
        adapter = AlpacaBrokerAdapter(mode=args.mode)
    else:
        adapter = ShioajiBrokerAdapter(mode=args.mode, twd_per_usd=limits.twd_per_usd)

    snap = adapter.get_account_snapshot()
    ranked = _load_ranked_signals(market, args.learned_weights, args.offline_demo, args.signals_csv)
    run_date = dt.date.today().isoformat()

    plan = build_buy_only_plan(
        ranked,
        market=market,
        cash_native=args.cash,
        current_positions_usd=snap.get("positions_usd", {}),
        account_equity_usd=float(snap.get("equity_usd", 0.0)),
        broker=args.broker,
        run_date=run_date,
        limits=limits,
        top_n=args.top_n,
    )

    report = execute_order_plan(
        plan,
        adapter,
        limits,
        run_date=run_date,
        ledger_path=args.ledger,
        audit_dir=args.audit_dir,
        kill_switch_path=args.kill_switch,
    )

    print(f"=== Snowball Broker Execution | {args.broker.upper()} | mode={args.mode.upper()} | date={run_date} ===")
    print(f"Summary: {report['summary']}")
    for item in report["executed"]:
        o = item["order"]
        print(f"  [OK]  {o['side'].upper():<4} {o['ticker']:<10} qty={o['qty']:<8} lim={o['currency']} {o['limit_price']:<8} (~${o['notional_usd']:.0f}) | {o['reason']}")
    for item in report["rejected"]:
        o = item["order"]
        print(f"  [REJ] {o['side'].upper():<4} {o['ticker']:<10} qty={o['qty']:<8} | {item['reason']}")
    print(f"Audit log saved: {report['audit_path']}")


if __name__ == "__main__":
    main()
