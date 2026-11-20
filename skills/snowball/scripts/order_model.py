"""Order model shared by the Snowball execution engine and broker adapters.

Constants, immutable risk limits, the normalized `OrderIntent`, price/tick helpers,
the deterministic client-order-id and the live-mode safety lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os

import numpy as np

import exec_safety as es

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


def make_client_order_id(run_date: str, broker: str, ticker: str, side: str, *, market: str) -> str:
    """Deterministic idempotency key: one order per (exchange trade date, broker scope, symbol, side).

    - Quantity is deliberately NOT part of the key, so a same-day re-run after a price or cash
      change cannot mint a fresh id and slip past the duplicate guard.
    - The symbol is normalized (`2330.TW` == `2330`, `BRK-B` == `BRK.B`) so differently spelled
      signal files cannot mint a fresh id either.
    - Pass a mode-qualified `broker` (e.g. `alpaca:live`) so paper and live never share ids.
    """
    raw = f"{run_date}|{broker.lower()}|{es.normalize_symbol(ticker, market)}|{side.lower()}"
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
