#!/usr/bin/env python3
"""
Prediction Tracker & Brier Score / BSS Evaluator (CLI).

Evaluates probabilistic market or stock forecasts stored in a CSV file,
computes excess return vs. benchmark, binary outcome, Brier Score, and
Brier Skill Score (BSS vs. 0.25 random coin-flip baseline).

Usage:
    python score_predictions.py --csv ./my_predictions.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def _pick(row: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k]).strip()
    return default


def evaluate_predictions(csv_path: Path) -> dict:
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    updated = False
    settled = 0
    hits = 0
    brier_sum = 0.0

    print("=" * 72)
    print("Snowball Prediction Calibration & Brier Score Summary")
    print("=" * 72)

    for row in rows:
        pred_id = _pick(row, "prediction_id", "預測ID", default="?")
        target = _pick(row, "target", "標的名稱", default="?")
        direction = _pick(row, "direction", "方向", default="看漲")
        prob = float(_pick(row, "probability", "預測機率", default="0.5"))
        bench = _pick(row, "benchmark", "對照基準", default="BENCH")
        asset_ret_str = _pick(row, "asset_return_pct", "標的實際報酬(%)")
        bench_ret_str = _pick(row, "benchmark_return_pct", "基準實際報酬(%)")

        if asset_ret_str != "" and bench_ret_str != "":
            asset_ret = float(asset_ret_str)
            bench_ret = float(bench_ret_str)
            excess = asset_ret - bench_ret
            is_bull = direction.lower() in ("long", "bull", "outperform", "up", "看漲")
            outcome = 1 if (excess > 0 if is_bull else excess < 0) else 0
            brier = (prob - outcome) ** 2

            for k, v in [
                ("excess_return_pct", f"{excess:+.2f}"),
                ("超額報酬(%)", f"{excess:+.2f}"),
                ("outcome", str(outcome)),
                ("結果(1=對/0=錯)", str(outcome)),
                ("brier_score", f"{brier:.4f}"),
                ("Brier_Score", f"{brier:.4f}"),
            ]:
                if k in fieldnames:
                    row[k] = v

            updated = True
            settled += 1
            hits += outcome
            brier_sum += brier
            status_str = f"HIT (Brier={brier:.4f})" if outcome == 1 else f"MISS (Brier={brier:.4f})"
        else:
            status_str = "PENDING"

        print(f"[{pred_id}] {direction} | {target:<16} | p={prob*100:4.1f}% | vs {bench:<6} | {status_str}")

    if updated:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    hit_rate = (hits / settled * 100.0) if settled > 0 else 0.0
    mean_brier = (brier_sum / settled) if settled > 0 else 0.0
    bss = (1.0 - mean_brier / 0.25) if settled > 0 else 0.0

    print("-" * 72)
    print(f"Total: {len(rows)} | Settled: {settled} | Pending: {len(rows) - settled}")
    if settled > 0:
        print(f"Hit Rate: {hit_rate:.1f}% ({hits}/{settled}) | Mean Brier: {mean_brier:.4f} | BSS vs 0.25: {bss:+.4f}")
    else:
        print("No settled rows yet. Fill in actual asset and benchmark returns and re-run.")
    print("=" * 72)
    return {"total": len(rows), "settled": settled, "hits": hits, "mean_brier": mean_brier, "bss": bss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score probabilistic forecasts and compute Brier Score / BSS.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to predictions CSV file.")
    args = parser.parse_args()
    evaluate_predictions(args.csv)


if __name__ == "__main__":
    main()
