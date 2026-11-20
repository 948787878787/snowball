"""Pure, network-free pre-trade safety helpers used by `broker_exec.py`.

Kept separate so every live-money guard is unit-testable without a broker:
- exchange-local trade dates (US session crosses midnight in Taipei),
- broker-agnostic symbol normalization (BRK-B vs BRK.B, 8299.TWO vs 8299),
- high-water-mark drawdown math,
- deployable cash after unsettled / pending buys,
- train/serve parity guard for the scoring model.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

EXCHANGE_TZ: Dict[str, str] = {"US": "America/New_York", "TW": "Asia/Taipei"}
FUNDAMENTAL_PILLARS: Tuple[str, ...] = ("norm_val", "quality", "fund")
FUNDAMENTAL_WEIGHT_TOLERANCE = 0.01
TW_BOARD_LOT = 1000
SHIOAJI_TAG_LEN = 6


def exchange_trade_date(market: str, now_utc: Optional[dt.datetime] = None) -> str:
    """Return the calendar date in the exchange's own timezone (ISO format)."""
    tz_name = EXCHANGE_TZ.get(market.upper())
    if tz_name is None:
        raise ValueError(f"Unknown market '{market}'. Expected one of {sorted(EXCHANGE_TZ)}.")
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware.")
    return now.astimezone(ZoneInfo(tz_name)).date().isoformat()


def normalize_symbol(ticker: str, market: str) -> str:
    """Canonical key shared by signals, broker positions, open orders and risk checks.

    TW: bare numeric code (`2330.TW`, `8299.TWO` -> `2330`, `8299`).
    US: Alpaca/exchange class-share format (`BRK-B` -> `BRK.B`).
    """
    sym = str(ticker).strip().upper()
    if market.upper() == "TW":
        for suffix in (".TWO", ".TW"):
            if sym.endswith(suffix):
                return sym[: -len(suffix)]
        return sym
    return sym.replace("-", ".")


def drawdown_from_hwm(equity_usd: float, hwm_usd: Optional[float]) -> Tuple[float, float]:
    """Return `(drawdown_pct, new_hwm)`; raises on non-positive equity (fail closed)."""
    eq = float(equity_usd)
    if not eq > 0.0:
        raise ValueError(f"Account equity must be positive to evaluate drawdown (got {equity_usd!r}).")
    new_hwm = eq if hwm_usd is None or not hwm_usd > 0.0 else max(float(hwm_usd), eq)
    return eq / new_hwm - 1.0, new_hwm


def deployable_cash_usd(requested_usd: float, available_usd: float, pending_buy_usd: float) -> float:
    """Cash that may be committed now: never above settled cash minus already-pending buys."""
    room = float(available_usd) - max(float(pending_buy_usd), 0.0)
    return max(0.0, min(float(requested_usd), room))


def tw_lot_split(shares: int) -> Tuple[int, str]:
    """Round a Taiwan share count to one valid order: whole board lots, or a 1-999 share odd lot."""
    n = int(shares)
    if n >= TW_BOARD_LOT:
        return (n // TW_BOARD_LOT) * TW_BOARD_LOT, "Common"
    return max(n, 0), "IntradayOdd"


def shioaji_custom_tag(client_order_id: str) -> str:
    """6-char Shioaji `custom_field` derived from the order's hash (unique per order per day)."""
    digest = client_order_id.rsplit("_", 1)[-1]
    tag = "".join(ch for ch in digest if ch.isalnum()).upper()[:SHIOAJI_TAG_LEN]
    if len(tag) != SHIOAJI_TAG_LEN:
        raise ValueError(f"client_order_id '{client_order_id}' cannot produce a {SHIOAJI_TAG_LEN}-char tag.")
    return tag


def assert_signal_parity(
    mode: str,
    signal_source: str,
    weights: Mapping[str, float],
    *,
    accept_skew: bool,
) -> None:
    """Refuse to route orders from a scoring path that differs from the backtested one.

    `signal_source="neutral_fundamentals"` means the value / quality / growth pillars are
    constant zero at inference, so any weight on them silently turns the model into an
    untested price-only ranker.
    """
    if mode.lower() == "dry_run" or accept_skew or signal_source != "neutral_fundamentals":
        return
    fund_w = sum(float(weights.get(k, 0.0)) for k in FUNDAMENTAL_PILLARS)
    if fund_w > FUNDAMENTAL_WEIGHT_TOLERANCE:
        raise RuntimeError(
            f"Signals were scored with neutral (all-zero) fundamentals while {fund_w:.0%} of the model "
            f"weight sits on fundamental pillars {FUNDAMENTAL_PILLARS}. This is a model that was never "
            "backtested. Pass --signals-csv from the validated pipeline, or --accept-signal-skew to override."
        )
