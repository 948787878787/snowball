#!/usr/bin/env python3
"""
Generic v5 Weekly Portfolio & Entry-Timing Monitor (Zero-PII).

Reads any user-supplied local JSON portfolio file (see references/portfolio-template.json),
computes latest v5 multi-factor rankings, cash-flow rebalancing priorities, and
overweight winner-trim alerts without hardcoding any personal financial data.

Usage:
    python monitor_portfolio.py --portfolio path/to/my_portfolio.json
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import data
from backtest_v3 import load_panels
from backtest_v5 import build_v5_panels


def main() -> None:
    parser = argparse.ArgumentParser(description="v5 Portfolio & Cash-Flow Priority Monitor")
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=SCRIPT_DIR.parent / "references" / "portfolio-template.json",
        help="Path to local portfolio JSON file (defaults to example template).",
    )
    args = parser.parse_args()

    if not args.portfolio.exists():
        print(f"Portfolio file not found: {args.portfolio}")
        sys.exit(1)

    cfg = json.loads(args.portfolio.read_text(encoding="utf-8"))
    fx = float(cfg.get("currency_fx_usd_twd", 32.5))
    w_target = float(cfg.get("satellite_target_weight", 0.10))
    max_price = float(cfg.get("max_odd_lot_price", 500.0))
    holdings = cfg.get("holdings", [])
    watchlist = cfg.get("watchlist", [])

    p = load_panels()
    v5 = build_v5_panels(p)
    close_df = data.load_prices()["close"]
    val_df = data.load_valuation().sort_values("date").groupby("stock_id").last()

    latest_daily_dt = close_df.index[-1]
    latest_panel_dt = v5["v5_income"].index[-1]
    s_v5_inc = v5["v5_income"].loc[latest_panel_dt].dropna().sort_values(ascending=False)
    s_v5_core = v5["v5_core"].loc[latest_panel_dt].dropna().sort_values(ascending=False)
    rank_inc = {sid: idx + 1 for idx, sid in enumerate(s_v5_inc.index)}
    rank_core = {sid: idx + 1 for idx, sid in enumerate(s_v5_core.index)}
    total_ranked = len(s_v5_inc)

    print(f"Latest Price Date: {latest_daily_dt.strftime('%Y-%m-%d')}")
    print(f"Latest v5 Model Date: {latest_panel_dt.strftime('%Y-%m-%d')} ({total_ranked} stocks)")
    print("")

    # 1. Portfolio valuation & winner-trim check
    mvs_twd = {}
    total_twd = 0.0
    evaluated = []
    for h in holdings:
        code = str(h["code"])
        mkt = str(h.get("market", "TW")).upper()
        shares = float(h["shares"])
        cost = float(h["cost"])
        if code in close_df.columns and not close_df[code].dropna().empty:
            price = float(close_df[code].dropna().iloc[-1])
        else:
            price = float(h.get("fallback_price", cost))
        mv = price * shares * (fx if mkt == "US" else 1.0)
        mvs_twd[code] = mv
        total_twd += mv
        evaluated.append((h, price, mv))

    print("1. Portfolio Status & Overweight Winner-Trim Check")
    for h, price, mv in evaluated:
        code = str(h["code"])
        cost = float(h["cost"])
        wt = mv / total_twd if total_twd > 0 else 0.0
        pnl_pct = (price - cost) / cost * 100.0 if cost > 0 else 0.0
        target_sell = h.get("target_sell")
        if target_sell is not None and price >= float(target_sell):
            status = f"TARGET HIT (>= {target_sell}) -> Trim/Rotate Profit"
        elif wt > 2.0 * w_target and price > cost:
            status = f"OVERWEIGHT WINNER ({wt*100:.1f}% > {2*w_target*100:.0f}%) -> Trim Excess Profit"
        elif price >= cost:
            status = "In Profit (Hold / Monitor)"
        else:
            status = "Core DCA" if h.get("role") == "core_dca" else "Frozen (No New Adds)"
        print(
            f"  [{h.get('market', 'TW')}] {code} {h.get('name', '')}: "
            f"Px={price:.2f} | Cost={cost:.2f} | Wt={wt*100:.1f}% | PnL={pnl_pct:+.2f}% | {status}"
        )

    print("")
    print("2. Watchlist v5 Scores & Cash-Flow Rebalancing Priority")
    for item in watchlist:
        code = str(item["code"])
        if code not in close_df.columns or close_df[code].dropna().empty:
            print(f"  {code} {item.get('name', '')}: Not in 84-stock v5 universe")
            continue
        price = float(close_df[code].dropna().iloc[-1])
        dy = float(val_df.loc[code, "dividend_yield"]) if code in val_df.index else 0.0
        per = float(val_df.loc[code, "PER"]) if code in val_df.index else 0.0
        sc = float(s_v5_inc.get(code, 0.0))
        rk_i = rank_inc.get(code, -1)
        rk_c = rank_core.get(code, -1)
        w_now = mvs_twd.get(code, 0.0) / total_twd if total_twd > 0 else 0.0
        prio = sc + 1.0 * (w_target - w_now) / w_target
        zone = "Top 20% Buy Zone" if 0 < rk_i <= int(total_ranked * 0.20) else ("Top 40% Buffer" if 0 < rk_i <= int(total_ranked * 0.40) else "Watch/Wait")
        print(
            f"  {code} {item.get('name', '')}: Px={price:.2f} | Yield={dy:.2f}% | PER={per:.1f} | "
            f"v5={sc:+.3f} (#{rk_i}/{total_ranked}, core #{rk_c}) | Priority={prio:+.2f} | {zone}"
        )

    print("")
    print(f"3. Top 10 v5_income Candidates (Price <= {max_price:.0f} TWD)")
    shown = 0
    for sid, sc in s_v5_inc.items():
        price = float(close_df[sid].dropna().iloc[-1]) if sid in close_df.columns else 99999.0
        if price > max_price:
            continue
        dy = float(val_df.loc[sid, "dividend_yield"]) if sid in val_df.index else 0.0
        per = float(val_df.loc[sid, "PER"]) if sid in val_df.index else 0.0
        w_now = mvs_twd.get(sid, 0.0) / total_twd if total_twd > 0 else 0.0
        prio = float(sc) + 1.0 * (w_target - w_now) / w_target
        print(f"  #{rank_inc[sid]:02d} {sid}: Px={price:.2f} | Yield={dy:.2f}% | PER={per:.1f} | v5={sc:+.3f} | Priority={prio:+.2f}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
