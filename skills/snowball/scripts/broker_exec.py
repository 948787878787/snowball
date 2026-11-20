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
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import exec_safety as es  # noqa: E402  (needs SCRIPT_DIR on sys.path)
from broker_adapters import (  # noqa: E402,F401  (re-exported for callers and tests)
    AlpacaBrokerAdapter,
    BaseBrokerAdapter,
    DryRunAdapter,
    ShioajiBrokerAdapter,
)
from order_model import (  # noqa: E402,F401  (re-exported for callers and tests)
    ALPACA_LIVE_URL,
    ALPACA_PAPER_URL,
    DEFAULT_TWD_PER_USD,
    LIVE_UNLOCK_ENV,
    LIVE_UNLOCK_PHRASE,
    OrderIntent,
    RiskLimits,
    assert_mode_allowed,
    compute_limit_price,
    make_client_order_id,
    tw_round_tick,
)

UNKNOWN_SUBMIT_STATUS = "UNKNOWN_NEEDS_RECONCILE"

__all__ = [
    "ALPACA_LIVE_URL", "ALPACA_PAPER_URL", "DEFAULT_TWD_PER_USD", "LIVE_UNLOCK_ENV", "LIVE_UNLOCK_PHRASE",
    "AlpacaBrokerAdapter", "BaseBrokerAdapter", "DryRunAdapter", "ShioajiBrokerAdapter",
    "IdempotencyLedger", "OrderIntent", "RiskLimits", "UNKNOWN_SUBMIT_STATUS",
    "assert_mode_allowed", "build_buy_only_plan", "compute_limit_price", "execute_order_plan",
    "make_client_order_id", "tw_round_tick", "validate_order_plan",
]


class IdempotencyLedger:
    """Atomic JSON ledger tracking submitted client_order_ids and daily notional."""

    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Any] = {"orders": {}, "daily_notional_usd": {}, "equity_hwm_usd": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Execution ledger {self.path} is unreadable ({exc}). Refusing to run: an empty ledger "
                    "would disable duplicate-order protection. Repair or move the file, then reconcile with the broker."
                ) from exc
            if not isinstance(loaded, dict):
                raise RuntimeError(f"Execution ledger {self.path} has unexpected structure; refusing to run.")
            self._data.update(loaded)
            self._data.setdefault("equity_hwm_usd", {})

    def has_order(self, client_order_id: str) -> bool:
        return client_order_id in self._data.get("orders", {})

    def daily_notional(self, run_date: str) -> float:
        return float(self._data.get("daily_notional_usd", {}).get(run_date, 0.0))

    def drawdown(self, broker: str, equity_usd: float, persist: bool) -> float:
        """Drawdown vs the persisted equity high-water mark; updates the HWM when `persist`.

        Note: withdrawals register as drawdown (fails safe); deposits raise the HWM.
        """
        hwm_map = self._data.setdefault("equity_hwm_usd", {})
        dd, new_hwm = es.drawdown_from_hwm(equity_usd, hwm_map.get(broker))
        if persist and new_hwm != hwm_map.get(broker):
            hwm_map[broker] = round(new_hwm, 2)
            self._flush()
        return dd

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


@dataclass(frozen=True)
class _ValidationContext:
    """Immutable inputs shared by every per-order pre-trade check."""

    limits: RiskLimits
    account_equity_usd: float
    positions_by_market: Dict[str, Dict[str, float]]
    pending_by_market: Dict[str, Dict[str, float]]
    ledger: Optional[IdempotencyLedger]
    broker_submitted_keys: Set[str]
    dedupe_key_fn: Callable[[OrderIntent], str]


def _normalize_by_market(values_usd: Dict[str, float], markets: Set[str]) -> Dict[str, Dict[str, float]]:
    """Re-key a {symbol: usd} map per market so `BRK-B`/`BRK.B` and `8299.TWO`/`8299` collide."""
    out: Dict[str, Dict[str, float]] = {}
    for mkt in markets:
        acc: Dict[str, float] = {}
        for sym, val in values_usd.items():
            key = es.normalize_symbol(sym, mkt)
            acc[key] = acc.get(key, 0.0) + float(val)
        out[mkt] = acc
    return out


def _static_order_reason(o: OrderIntent, ctx: _ValidationContext) -> Optional[str]:
    """Checks that do not depend on earlier approvals in the same run."""
    lim = ctx.limits
    if o.qty <= 0 or o.limit_price <= 0:
        return "INVALID_QTY_OR_PRICE"
    if o.notional_usd < lim.min_order_notional_usd:
        return f"BELOW_MIN_NOTIONAL (${o.notional_usd:.2f} < ${lim.min_order_notional_usd:.2f})"
    if o.notional_usd > lim.max_single_order_usd:
        return f"EXCEEDS_MAX_SINGLE_ORDER (${o.notional_usd:.2f} > ${lim.max_single_order_usd:.2f})"
    if ctx.ledger is not None and o.client_order_id and ctx.ledger.has_order(o.client_order_id):
        return f"DUPLICATE_CLIENT_ORDER_ID ({o.client_order_id})"
    if o.client_order_id and ctx.dedupe_key_fn(o) in ctx.broker_submitted_keys:
        return f"ALREADY_AT_BROKER ({ctx.dedupe_key_fn(o)})"
    if o.side == "buy":
        if not ctx.account_equity_usd > 0.0:
            return f"EQUITY_UNKNOWN (account equity {ctx.account_equity_usd!r}; refusing buys)"
        mkt = o.market.upper()
        key = es.normalize_symbol(o.ticker, mkt)
        cur_val = ctx.positions_by_market.get(mkt, {}).get(key, 0.0) + ctx.pending_by_market.get(mkt, {}).get(key, 0.0)
        post_weight = (cur_val + o.notional_usd) / ctx.account_equity_usd
        if post_weight > lim.max_single_position_weight + 1e-9:
            return f"EXCEEDS_MAX_POSITION_WEIGHT ({post_weight:.1%} > {lim.max_single_position_weight:.1%}, incl. open orders)"
    return None


def _reject_all(orders: Sequence[OrderIntent], reason: str, keep_sells: bool) -> Tuple[List[OrderIntent], List[Dict[str, Any]]]:
    approved = [o for o in orders if keep_sells and o.side != "buy"]
    rejected = [{"order": asdict(o), "reason": reason} for o in orders if not (keep_sells and o.side != "buy")]
    return approved, rejected


def validate_order_plan(
    orders: Sequence[OrderIntent],
    limits: RiskLimits,
    *,
    account_equity_usd: float,
    current_positions_usd: Dict[str, float],
    run_date: str,
    ledger: Optional[IdempotencyLedger] = None,
    kill_switch_path: Optional[Path] = None,
    account_drawdown_pct: Optional[float] = None,
    pending_buy_usd: Optional[Dict[str, float]] = None,
    broker_submitted_keys: Optional[Set[str]] = None,
    available_cash_usd: Optional[float] = None,
    dedupe_key_fn: Optional[Callable[[OrderIntent], str]] = None,
) -> Tuple[List[OrderIntent], List[Dict[str, Any]]]:
    """Validate every proposed order against all pre-trade risk guards (fail closed).

    - `account_drawdown_pct=None` means drawdown is unknown -> all buys rejected.
    - `pending_buy_usd` (open broker orders) counts toward position weight and cash.
    - `available_cash_usd`, when given, caps cumulative buy notional (no margin / no default settlement).

    Returns `(approved_orders, rejected_records)`.
    """
    if kill_switch_path is not None and kill_switch_path.exists():
        return _reject_all(orders, f"KILL_SWITCH_ACTIVE ({kill_switch_path})", keep_sells=False)
    if account_drawdown_pct is None:
        return _reject_all(orders, "DRAWDOWN_UNKNOWN (no high-water mark / equity; refusing buys)", keep_sells=True)
    if account_drawdown_pct <= limits.max_drawdown_halt_pct:
        reason = f"DRAWDOWN_HALT ({account_drawdown_pct:.1%} <= {limits.max_drawdown_halt_pct:.1%})"
        return _reject_all(orders, reason, keep_sells=True)

    markets = {o.market.upper() for o in orders}
    pending = dict(pending_buy_usd or {})
    ctx = _ValidationContext(
        limits=limits,
        account_equity_usd=float(account_equity_usd),
        positions_by_market=_normalize_by_market(current_positions_usd, markets),
        pending_by_market=_normalize_by_market(pending, markets),
        ledger=ledger,
        broker_submitted_keys=set(broker_submitted_keys or set()),
        dedupe_key_fn=dedupe_key_fn or (lambda o: o.client_order_id),
    )
    cash_left = None if available_cash_usd is None else float(available_cash_usd) - sum(max(v, 0.0) for v in pending.values())
    running_notional = ledger.daily_notional(run_date) if ledger is not None else 0.0

    approved: List[OrderIntent] = []
    rejected: List[Dict[str, Any]] = []
    for o in orders:
        reason = _static_order_reason(o, ctx)
        if reason is None and len(approved) >= limits.max_orders_per_run:
            reason = f"MAX_ORDERS_PER_RUN ({limits.max_orders_per_run})"
        if reason is None and running_notional + o.notional_usd > limits.max_daily_notional_usd:
            reason = f"EXCEEDS_DAILY_NOTIONAL (${running_notional + o.notional_usd:.2f} > ${limits.max_daily_notional_usd:.2f})"
        if reason is None and o.side == "buy" and cash_left is not None and o.notional_usd > cash_left + 1e-9:
            reason = f"INSUFFICIENT_CASH (${o.notional_usd:.2f} > available ${max(cash_left, 0.0):.2f} after open orders)"
        if reason is not None:
            rejected.append({"order": asdict(o), "reason": reason})
            continue
        running_notional += o.notional_usd
        if o.side == "buy" and cash_left is not None:
            cash_left -= o.notional_usd
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
    held_usd = _normalize_by_market(current_positions_usd, {mkt})[mkt]
    priorities = []
    for _, row in cands.iterrows():
        sym = str(row["ticker"])
        cur_w = float(held_usd.get(es.normalize_symbol(sym, mkt), 0.0)) / eq
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
            # One valid order per name: whole board lots, or a 1-999 share intraday odd lot
            shares, lot_type = es.tw_lot_split(int(np.floor(slot_native / lim_px)))
            if shares <= 0:
                continue
            qty = float(shares)
        else:
            qty = round(slot_native / lim_px, 4) if allow_fractional_us else float(int(np.floor(slot_native / lim_px)))
            if qty <= 0:
                continue
            lot_type = ""

        notional_usd = qty * lim_px * fx_to_usd
        cid = make_client_order_id(run_date, broker, sym, "buy", market=mkt)
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


def _fetch_order_state(adapter: BaseBrokerAdapter, trade_date: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return `(state, error)`; `state is None` means broker-side state is unknown (fail closed)."""
    try:
        raw = adapter.get_order_state(trade_date)
    except Exception as exc:  # network, auth, NotImplementedError: all mean "unknown"
        return None, f"{type(exc).__name__}: {exc}"
    state = {
        "pending_buy_usd": {str(k): float(v) for k, v in dict(raw.get("pending_buy_usd", {})).items()},
        "submitted_keys": set(raw.get("submitted_keys", set())),
    }
    return state, ""


def _resolve_drawdown(ledger: IdempotencyLedger, broker: str, equity_usd: float, persist: bool) -> Optional[float]:
    try:
        return ledger.drawdown(broker, equity_usd, persist=persist)
    except ValueError:
        return None  # non-positive / missing equity -> unknown -> buys rejected


def _submit_approved(
    approved: Sequence[OrderIntent],
    adapter: BaseBrokerAdapter,
    ledger: IdempotencyLedger,
    run_date: str,
    is_real: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Submit approved orders; a ledger write failure halts the run (duplicate protection is gone)."""
    executed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for idx, o in enumerate(approved):
        submit_exc: Optional[Exception] = None
        try:
            resp = adapter.submit_limit_order(o)
        except Exception as exc:  # timeout / reset: the broker may or may not hold the order
            submit_exc, resp = exc, {"status": UNKNOWN_SUBMIT_STATUS, "id": ""}
        try:
            if is_real:
                ledger.record(o, run_date, resp)
        except OSError as exc:
            errors.append({
                "order": asdict(o),
                "error": f"LEDGER_WRITE_FAILED ({exc}); halting run, reconcile with broker",
                "ledger_status": "NOT_RECORDED",
            })
            errors.extend(
                {"order": asdict(rest), "error": "SKIPPED_AFTER_LEDGER_FAILURE", "ledger_status": "NOT_SUBMITTED"}
                for rest in approved[idx + 1:]
            )
            break
        if submit_exc is not None:
            errors.append({
                "order": asdict(o),
                "error": f"{type(submit_exc).__name__}: {submit_exc}",
                "ledger_status": UNKNOWN_SUBMIT_STATUS,
            })
        else:
            executed.append({"order": asdict(o), "response": resp})
    return executed, errors


def execute_order_plan(
    orders: Sequence[OrderIntent],
    adapter: BaseBrokerAdapter,
    limits: RiskLimits,
    *,
    run_date: str,
    ledger_path: Path,
    audit_dir: Path,
    kill_switch_path: Optional[Path] = None,
    account_drawdown_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Reconcile broker state, validate, execute (or dry-run), record ledger and write a JSON audit log.

    - Drawdown defaults to the ledger's persisted equity high-water mark (never a silent 0.0).
    - Open/today's broker orders feed the duplicate, position-weight and cash guards.
    - A submit that raises (timeout, connection reset) is recorded as UNKNOWN so it is never
      blindly resubmitted the same trade date; reconcile it with the broker manually.
    """
    snap = adapter.get_account_snapshot()
    ledger = IdempotencyLedger(ledger_path)
    is_real = adapter.name != "dry_run"
    equity = float(snap.get("equity_usd", 0.0))
    drawdown = (
        account_drawdown_pct
        if account_drawdown_pct is not None
        else _resolve_drawdown(ledger, adapter.ledger_scope, equity, persist=is_real)
    )
    state, state_err = _fetch_order_state(adapter, run_date)

    if state is None:
        approved: List[OrderIntent] = []
        rejected = [
            {"order": asdict(o), "reason": f"ORDER_STATE_UNKNOWN ({state_err}); reconcile with broker before trading"}
            for o in orders
        ]
    else:
        approved, rejected = validate_order_plan(
            orders,
            limits,
            account_equity_usd=equity,
            current_positions_usd=dict(snap.get("positions_usd", {})),
            run_date=run_date,
            ledger=ledger,
            kill_switch_path=kill_switch_path,
            account_drawdown_pct=drawdown,
            pending_buy_usd=state["pending_buy_usd"],
            broker_submitted_keys=state["submitted_keys"],
            available_cash_usd=float(snap.get("available_cash_usd", 0.0)),
            dedupe_key_fn=adapter.order_dedupe_key,
        )

    executed, errors = _submit_approved(approved, adapter, ledger, run_date, is_real)

    report = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_date": run_date,
        "broker": adapter.name,
        "account_snapshot": snap,
        "account_drawdown_pct": drawdown,
        "order_state": None if state is None else {
            "pending_buy_usd": state["pending_buy_usd"],
            "submitted_keys": sorted(state["submitted_keys"]),
        },
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
    ts_tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    audit_path = audit_dir / f"order_audit_{adapter.name}_{ts_tag}.json"
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report["audit_path"] = str(audit_path)
    return report


def _load_ranked_signals(
    market: str,
    learned_weights_path: Optional[Path],
    offline_demo: bool,
    signals_csv: Optional[Path],
) -> Tuple[pd.DataFrame, str, Dict[str, float]]:
    """Return `(ranked_df, signal_source, pillar_weights)` for the train/serve parity guard."""
    from macro_micro import BASELINE_V6_WEIGHTS

    if signals_csv is not None:
        return pd.read_csv(signals_csv), "csv", {}

    import global_ranker as gr
    from macro_micro import compute_macro_regime_panel, download_macro_series, make_offline_macro_data

    groups, tickers, benchmark, _, _, _, convert_usd = gr.resolve_universe_selection(market=market)
    if offline_demo:
        close_d, open_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
        macro_d = make_offline_macro_data(close_d.index)
        source = "offline_demo"
    else:
        close_d, open_d, bench_d = gr.download_global_prices(tickers, benchmark, period="2y")
        if convert_usd:
            close_d, open_d = gr.convert_prices_to_usd(close_d, open_d, period="2y")
        macro_d = download_macro_series(period="2y")
        fund_df = gr.make_neutral_fundamentals(list(close_d.columns))
        source = "neutral_fundamentals"

    pf = gr.compute_global_price_factors(close_d, bench_d, groups, open_d=open_d)
    macro_panel = compute_macro_regime_panel(macro_d, pf["close_w"].index)
    latest_macro = macro_panel.iloc[-1].to_dict() if not macro_panel.empty else None
    learned_w = gr.load_learned_weights(learned_weights_path) if learned_weights_path else None
    ranked = gr.score_latest_global(pf, fund_df, groups, model="v6", macro_regime=latest_macro, learned_weights=learned_w)
    return ranked, source, dict(learned_w or BASELINE_V6_WEIGHTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball Safe Paper/Live Broker Execution Engine (Alpaca + Shioaji)")
    parser.add_argument("--broker", default="alpaca", choices=["alpaca", "shioaji"], help="Target broker adapter")
    parser.add_argument("--mode", default="dry_run", choices=["dry_run", "paper", "live"], help="Execution mode (default: dry_run)")
    parser.add_argument("--cash", type=float, default=1500.0, help="Max cash to deploy in broker native currency (USD for Alpaca, TWD for Shioaji); capped by settled cash minus open buys")
    parser.add_argument("--top-n", type=int, default=4, help="Max number of Top-20%% candidates to allocate cash across")
    parser.add_argument("--max-order-usd", type=float, default=1500.0, help="Hard cap per single order in USD")
    parser.add_argument("--max-daily-usd", type=float, default=5000.0, help="Hard daily total notional cap in USD")
    parser.add_argument("--max-weight", type=float, default=0.15, help="Hard single-stock portfolio weight ceiling")
    parser.add_argument("--signals-csv", type=Path, default=None, help="Optional pre-computed signals CSV (columns: ticker, close, v6_score)")
    parser.add_argument("--learned-weights", type=Path, default=None, help="Optional v6_learned_weights.json from Kaggle/Walk-Forward")
    parser.add_argument("--ledger", type=Path, default=Path("execution_ledger.json"), help="Idempotency ledger JSON path")
    parser.add_argument("--audit-dir", type=Path, default=Path("results"), help="Directory for order audit JSON logs")
    parser.add_argument("--kill-switch", type=Path, default=Path(".snowball_kill_switch"), help="Kill-switch file path (if present, halts all orders)")
    parser.add_argument("--offline-demo", action="store_true", help="Generate synthetic signals offline for end-to-end safety testing")
    parser.add_argument(
        "--accept-signal-skew",
        action="store_true",
        help="Allow paper/live orders from neutral-fundamental live scoring (differs from the backtested model)",
    )
    args = parser.parse_args()

    assert_mode_allowed(args.mode)
    if args.offline_demo and args.mode == "live":
        raise PermissionError("--offline-demo produces synthetic signals and can never be used with --mode live.")
    market = "TW" if args.broker == "shioaji" else "US"
    limits = RiskLimits(
        max_single_order_usd=args.max_order_usd,
        max_daily_notional_usd=args.max_daily_usd,
        max_single_position_weight=args.max_weight,
    )
    fx_to_usd = (1.0 / limits.twd_per_usd) if market == "TW" else 1.0

    if args.mode == "dry_run":
        cash_usd = args.cash * fx_to_usd
        adapter: BaseBrokerAdapter = DryRunAdapter(equity_usd=max(20000.0, cash_usd * 10.0), cash_usd=cash_usd)
    elif args.broker == "alpaca":
        adapter = AlpacaBrokerAdapter(mode=args.mode)
    else:
        adapter = ShioajiBrokerAdapter(mode=args.mode, twd_per_usd=limits.twd_per_usd)

    run_date = es.exchange_trade_date(market)
    ranked, signal_source, pillar_weights = _load_ranked_signals(market, args.learned_weights, args.offline_demo, args.signals_csv)
    es.assert_signal_parity(args.mode, signal_source, pillar_weights, accept_skew=args.accept_signal_skew)

    snap = adapter.get_account_snapshot()
    state, state_err = _fetch_order_state(adapter, run_date)
    pending_total_usd = sum(state["pending_buy_usd"].values()) if state is not None else float("inf")
    deploy_usd = es.deployable_cash_usd(args.cash * fx_to_usd, float(snap.get("available_cash_usd", 0.0)), pending_total_usd)
    cash_native = deploy_usd / fx_to_usd
    if cash_native + 1e-6 < args.cash:
        print(
            f"[CASH CAP] Requested {args.cash:,.2f} but only {cash_native:,.2f} is deployable "
            f"(settled cash minus open buys{'; order state unknown: ' + state_err if state is None else ''})."
        )

    plan = build_buy_only_plan(
        ranked,
        market=market,
        cash_native=cash_native,
        current_positions_usd=snap.get("positions_usd", {}),
        account_equity_usd=float(snap.get("equity_usd", 0.0)),
        broker=f"{args.broker}:{args.mode}",
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

    print(f"=== Snowball Broker Execution | {args.broker.upper()} | mode={args.mode.upper()} | trade_date={run_date} ===")
    print(f"Summary: {report['summary']} | drawdown={report['account_drawdown_pct']}")
    for item in report["executed"]:
        o = item["order"]
        print(f"  [OK]  {o['side'].upper():<4} {o['ticker']:<10} qty={o['qty']:<8} lim={o['currency']} {o['limit_price']:<8} (~${o['notional_usd']:.0f}) | {o['reason']}")
    for item in report["rejected"]:
        o = item["order"]
        print(f"  [REJ] {o['side'].upper():<4} {o['ticker']:<10} qty={o['qty']:<8} | {item['reason']}")
    for item in report["errors"]:
        o = item["order"]
        print(f"  [ERR] {o['side'].upper():<4} {o['ticker']:<10} | {item['error']} -> ledger={item['ledger_status']} (reconcile manually)")
    print(f"Audit log saved: {report['audit_path']}")


if __name__ == "__main__":
    main()
