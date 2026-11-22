"""Local transaction ledger for the core ETF account (CSV kept OUTSIDE the git repo).

Columns: date, type, symbol, shares, price, fee, tax, amount_twd, note
  deposit / withdraw / carry : amount_twd = money in / out (carry = hurdle carry paid to the manager)
  buy / sell                  : shares, price (native currency), fee, tax; amount_twd = total TWD cash
                                moved (required for USD symbols, optional for TWD ones)
  dividend / interest         : amount_twd = net cash received
  fee                         : amount_twd = other charge paid
Every cash change is derived from these rows; nothing is mutated in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from etf_config import is_usd

COLUMNS = ["date", "type", "symbol", "shares", "price", "fee", "tax", "amount_twd", "note"]
TYPES = {"deposit", "withdraw", "carry", "buy", "sell", "dividend", "interest", "fee"}
EXTERNAL_IN, EXTERNAL_OUT = {"deposit"}, {"withdraw", "carry"}


def init_ledger(path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the ledger is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)


def _cash_delta(row: pd.Series) -> float:
    t, amt = row["type"], row["amount_twd"]
    if t in ("buy", "sell"):
        if np.isfinite(amt):
            return -amt if t == "buy" else amt
        if is_usd(row["symbol"]):
            raise ValueError(f"{row['date'].date()} {row['symbol']}: USD trades need amount_twd")
        gross = row["shares"] * row["price"]
        return -(gross + row["fee"] + row["tax"]) if t == "buy" else gross - row["fee"] - row["tax"]
    if not np.isfinite(amt) or amt < 0:
        raise ValueError(f"{row['date'].date()} {t}: amount_twd must be a non-negative number")
    return amt if t in ("deposit", "dividend", "interest") else -amt


def _validate(led: pd.DataFrame) -> None:
    bad = sorted(set(led["type"]) - TYPES)
    if bad:
        raise ValueError(f"unknown ledger type(s): {bad}")
    trades = led[led["type"].isin(["buy", "sell"])]
    if (trades["shares"] <= 0).any() or (trades["price"] <= 0).any() or trades["symbol"].eq("").any():
        raise ValueError("buy/sell rows need symbol, shares > 0 and price > 0")
    signed = trades["shares"].where(trades["type"] == "buy", -trades["shares"])
    held = signed.groupby(trades["symbol"]).cumsum()
    if (held < 0).any():
        raise ValueError("a sell makes holdings negative")
    if (led["cash_delta"].cumsum() < -1e-6).any():
        raise ValueError("cash balance goes negative; record the deposit first")


def load_ledger(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"symbol": str, "note": str, "type": str})
    if raw.empty:
        raise ValueError(f"{path} is empty: add the first deposit row (date,deposit,,,,,,100000,...)")
    led = raw.reindex(columns=COLUMNS).copy()
    led["date"] = pd.to_datetime(led["date"])
    led["type"] = led["type"].str.strip().str.lower()
    led["symbol"] = led["symbol"].fillna("").str.strip()
    for c in ("shares", "price", "fee", "tax", "amount_twd"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led[["fee", "tax"]] = led[["fee", "tax"]].fillna(0.0)
    led = led.sort_values("date", kind="stable").reset_index(drop=True)
    bad = sorted(set(led["type"]) - TYPES)
    if bad:
        raise ValueError(f"unknown ledger type(s): {bad}")
    led["cash_delta"] = led.apply(_cash_delta, axis=1)
    _validate(led)
    return led


def cash_balance(led: pd.DataFrame, as_of) -> float:
    return float(led.loc[led["date"] <= pd.Timestamp(as_of), "cash_delta"].sum())


def positions(led: pd.DataFrame, as_of) -> Dict[str, int]:
    tr = led[(led["date"] <= pd.Timestamp(as_of)) & led["type"].isin(["buy", "sell"])]
    signed = tr["shares"].where(tr["type"] == "buy", -tr["shares"])
    held = signed.groupby(tr["symbol"]).sum()
    return {s: int(round(v)) for s, v in held.items() if round(v) != 0}


def external_flows(led: pd.DataFrame) -> pd.Series:
    """Money entering (+) or leaving (-) the account by date; trading cash moves are internal."""
    ext = led[led["type"].isin(EXTERNAL_IN | EXTERNAL_OUT)]
    return ext.groupby("date")["cash_delta"].sum()


__all__ = ["COLUMNS", "TYPES", "init_ledger", "load_ledger", "cash_balance", "positions", "external_flows"]
