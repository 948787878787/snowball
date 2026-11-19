#!/usr/bin/env python3
"""
週報預測紀錄計分與校準工具 (Prediction Tracker & Brier Score Evaluator)
讀取並更新 `預測紀錄.csv`，自動計算相對基準超額報酬、命中率與 Brier Score。
"""

import csv
import sys
from pathlib import Path

CSV_PATH = Path(__file__).parent / "預測紀錄.csv"


def evaluate_predictions(csv_path: Path = CSV_PATH) -> None:
    if not csv_path.exists():
        print(f"找不到檔案：{csv_path}")
        sys.exit(1)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    updated = False
    total = len(rows)
    settled = 0
    hits = 0
    brier_sum = 0.0

    print("=" * 72)
    print("📊 週報預測紀錄與校準總覽")
    print("=" * 72)

    for row in rows:
        pred_id = row["預測ID"]
        target = row["標的名稱"]
        direction = row["方向"]
        prob = float(row["預測機率"])
        asset_ret_str = row.get("標的實際報酬(%)", "").strip()
        bench_ret_str = row.get("基準實際報酬(%)", "").strip()

        if asset_ret_str != "" and bench_ret_str != "":
            asset_ret = float(asset_ret_str)
            bench_ret = float(bench_ret_str)
            excess = asset_ret - bench_ret
            if direction == "看漲":
                outcome = 1 if excess > 0 else 0
            else:  # 看跌 (相對基準落後)
                outcome = 1 if excess < 0 else 0

            brier = (prob - outcome) ** 2
            row["超額報酬(%)"] = f"{excess:+.2f}"
            row["結果(1=對/0=錯)"] = str(outcome)
            row["Brier_Score"] = f"{brier:.4f}"
            updated = True

            settled += 1
            hits += outcome
            brier_sum += brier
            status_str = f"✅ 命中 (Brier={brier:.4f})" if outcome == 1 else f"❌ 落空 (Brier={brier:.4f})"
        else:
            status_str = f"⏳ 待結算 ({row['評估終點']})"

        print(f"[{pred_id}] {direction} | {target:<14} | 機率: {prob*100:4.1f}% | 基準: {row['對照基準']:<5} | {status_str}")

    if updated:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("-" * 72)
    print(f"總預測筆數：{total} 筆 | 已結算：{settled} 筆 | 待結算：{total - settled} 筆")
    if settled > 0:
        hit_rate = hits / settled * 100
        mean_brier = brier_sum / settled
        print(f"累計命中率：{hit_rate:.1f}% ({hits}/{settled})")
        print(f"平均 Brier Score：{mean_brier:.4f} (隨機猜測 50% 的基準為 0.2500，越低代表機率校準越準確)")
    else:
        print("目前尚無已結算預測。於評估終點後填入「標的實際報酬(%)」與「基準實際報酬(%)」再執行本腳本即可自動計分。")
    print("=" * 72)


if __name__ == "__main__":
    evaluate_predictions()
