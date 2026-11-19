#!/usr/bin/env python3
"""CLI tool to inspect and expand Snowball's global equity universe database.

Supports:
- Listing all markets, stock counts, and industry breakdowns
- Appending new tickers to any market/industry (built-in or user custom directory)
- Creating brand-new regional markets (e.g., KR, AU, CA, IN, SG)
- Bulk-importing tickers from a CSV file (`market,industry,ticker`)

Examples:
    python manage_universe.py --list
    python manage_universe.py --market US --show-industries
    python manage_universe.py --market US --industry software_internet --add APP,MDB,NET
    python manage_universe.py --market KR --name "South Korea (KRX)" --benchmark ^KS11 --currency KRW --industry semiconductors_hardware --add 005930.KS,000660.KS
    python manage_universe.py --import-csv ./extra_stocks.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from global_universe import (
    BUILTIN_UNIVERSES_DIR,
    DEFAULT_CUSTOM_UNIVERSES_DIR,
    MARKET_CONFIG,
    MARKET_INDUSTRY_LABELS,
    add_tickers_to_universe,
    get_market_groups,
    get_market_tickers,
    list_markets,
)


def cmd_list() -> None:
    print(f"{'Market':<8} {'Stocks':>7} {'Industries':>10} {'Min/Ind':>8} {'Benchmark':<12} {'Name'}")
    print("-" * 88)
    for mkt in list_markets():
        groups = get_market_groups(mkt)
        tickers = get_market_tickers(mkt)
        min_ind = min((len(v) for v in groups.values()), default=0)
        cfg = MARKET_CONFIG[mkt]
        print(f"{mkt:<8} {len(tickers):>7} {len(groups):>10} {min_ind:>8} {cfg['benchmark']:<12} {cfg['name']}")


def cmd_show_industries(market: str) -> None:
    mkt = market.upper()
    groups = get_market_groups(mkt)
    labels = MARKET_INDUSTRY_LABELS.get(mkt, {})
    print(f"=== {mkt} ({len(get_market_tickers(mkt))} stocks across {len(groups)} industries) ===")
    for ind, mems in groups.items():
        lbl = labels.get(ind, ind)
        preview = ", ".join(mems[:6]) + (" ..." if len(mems) > 6 else "")
        print(f"  {ind:<28} ({len(mems):>3}) | {lbl:<44} | {preview}")


def cmd_import_csv(csv_path: Path, target_dir: Path) -> None:
    grouped = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mkt = (row.get("market") or "").strip().upper()
            ind = (row.get("industry") or "").strip()
            ticker = (row.get("ticker") or "").strip()
            if mkt and ind and ticker:
                grouped[(mkt, ind)].append(ticker)

    total_added = 0
    for (mkt, ind), syms in grouped.items():
        path, added, total = add_tickers_to_universe(mkt, ind, syms, target_dir=target_dir)
        total_added += added
        print(f"[OK] {mkt}.{ind}: +{added} added (file total {total}) -> {path}")
    print(f"Import complete: {total_added} new tickers added.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and extend Snowball's global equity database.")
    parser.add_argument("--list", action="store_true", help="List all markets and stock counts")
    parser.add_argument("--market", default="", help="Market code (e.g., US, EU, JP, HK, TW, or a new code like KR)")
    parser.add_argument("--show-industries", action="store_true", help="Show industry breakdown for --market")
    parser.add_argument("--industry", default="", help="Industry key to append tickers into")
    parser.add_argument("--add", default="", help="Comma-separated Yahoo Finance tickers to add")
    parser.add_argument("--import-csv", type=Path, default=None, help="Bulk-import CSV with columns: market,industry,ticker")
    parser.add_argument("--custom", action="store_true", help="Write to custom_universes/ instead of built-in universes/")
    parser.add_argument("--name", default="", help="Optional market display name (when creating a new market)")
    parser.add_argument("--benchmark", default="", help="Optional benchmark ticker (when creating a new market)")
    parser.add_argument("--currency", default="", help="Optional base currency code (when creating a new market)")
    parser.add_argument("--cost", type=float, default=None, help="Optional round-trip transaction cost")
    args = parser.parse_args()

    target_dir = DEFAULT_CUSTOM_UNIVERSES_DIR if args.custom else BUILTIN_UNIVERSES_DIR

    if args.list:
        cmd_list()
        return

    if args.import_csv is not None:
        cmd_import_csv(args.import_csv, target_dir)
        return

    if args.market and args.show_industries:
        cmd_show_industries(args.market)
        return

    if args.market and args.industry and args.add:
        syms = [s.strip() for s in args.add.split(",") if s.strip()]
        path, added, total = add_tickers_to_universe(
            market=args.market,
            industry=args.industry,
            tickers=syms,
            target_dir=target_dir,
            benchmark=args.benchmark or None,
            currency=args.currency or None,
            name=args.name or None,
            round_trip_cost=args.cost,
        )
        print(f"[OK] Added {added} ticker(s) to {args.market.upper()}.{args.industry} (total in file: {total}) -> {path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
