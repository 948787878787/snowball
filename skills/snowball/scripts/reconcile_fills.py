"""Execution Fill Reconciliation, Implementation Shortfall & Buffer-Band Sell Planner (Snowball v8).

References:
- Perold (1988), *Journal of Portfolio Management*: Implementation Shortfall (arrival/ref price vs fill price).
- Garleanu & Pedersen (2013), *Journal of Finance*: No-trade buffer region for transaction-cost-optimal rebalancing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def reconcile_fills(
    fills_df: pd.DataFrame,
    *,
    log_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Compute Perold (1988) Implementation Shortfall (bps) for actual broker fills and append audit log."""
    if fills_df is None or fills_df.empty:
        return {"n_fills": 0, "mean_shortfall_bps": np.nan, "median_shortfall_bps": np.nan, "p95_shortfall_bps": np.nan, "total_notional": 0.0}

    df = fills_df.copy()
    side_sign = np.where(df["side"].astype(str).str.lower() == "buy", 1.0, -1.0)
    ref_px = df["ref_price"].astype(float).replace(0.0, np.nan)
    fill_px = df["fill_price"].astype(float)
    df["shortfall_bps"] = side_sign * ((fill_px - ref_px) / ref_px) * 10_000.0
    df["notional"] = fill_px * df["qty"].astype(float).abs()

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        prev = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        pd.concat([prev, df], ignore_index=True).to_csv(log_path, index=False)

    valid_sf = df["shortfall_bps"].dropna()
    return {
        "n_fills": int(len(df)),
        "mean_shortfall_bps": float(valid_sf.mean()) if len(valid_sf) else np.nan,
        "median_shortfall_bps": float(valid_sf.median()) if len(valid_sf) else np.nan,
        "p95_shortfall_bps": float(valid_sf.quantile(0.95)) if len(valid_sf) else np.nan,
        "total_notional": float(df["notional"].sum()),
    }


def build_buffer_sell_plan(
    ranked_df: pd.DataFrame,
    held_tickers: Sequence[str],
    *,
    hold_top_pct: float = 0.40,
    score_col: Optional[str] = None,
) -> List[str]:
    """Return ONLY currently held tickers that fell below the top `hold_top_pct` cross-sectional buffer."""
    if ranked_df is None or ranked_df.empty or not held_tickers:
        return []
    sc = score_col or next((c for c in ("v6_score", "v5_score", "score") if c in ranked_df.columns), ranked_df.columns[-1])
    ranked = ranked_df.dropna(subset=["ticker", sc]).sort_values(sc, ascending=False).reset_index(drop=True)
    cutoff = max(1, int(np.ceil(len(ranked) * float(hold_top_pct))))
    keep_zone = set(ranked["ticker"].iloc[:cutoff].astype(str).tolist())
    return [str(t) for t in held_tickers if str(t) not in keep_zone]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile broker fills and measure Implementation Shortfall (bps)")
    parser.add_argument("--fills-csv", type=Path, required=True, help="CSV with trade_date,ticker,side,ref_price,fill_price,qty")
    parser.add_argument("--log-csv", type=Path, default=Path("results/fill_slippage_log.csv"), help="Cumulative slippage log CSV")
    args = parser.parse_args()
    summary = reconcile_fills(pd.read_csv(args.fills_csv), log_path=args.log_csv)
    print("Fill Reconciliation Summary:", summary)


if __name__ == "__main__":
    main()
