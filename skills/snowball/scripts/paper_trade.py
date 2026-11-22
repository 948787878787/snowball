"""Snowball v8 weekly paper-trading recorder (T3). Records model signals only; never places orders.

Each run takes the latest signal date in the v8 panel and writes an append-only snapshot of every
universe member's score and percentile for each pre-registered arm, stamped with the config SHA-256,
git commit and timestamps. Performance is evaluated later by replaying these snapshots against prices
that were not known when the snapshot was written.

Usage (log dir OUTSIDE the git repo):
  python paper_trade.py --panel ../../../../kaggle_dataset/v8/snowball_tw_v8.parquet \
      --config ../../../configs/v8_frozen.json --model ../../../../paper_trading/models/v8_ridge.json \
      --log-dir ../../../../paper_trading --watch 2915,2881,2308,2204,2646 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

import v8_composite as vc
import v8_model as vm

ARM_SCORES = {
    "P1_v8_blend": {"v8_zero": 0.5, "v8_ridge": 0.5},
    "P2_v6_prior": {"pred_baseline_v6_prior": 1.0},
    "P3_v8_zero": {"v8_zero": 1.0},
}
TOP_N_REPORT = 10


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def score_snapshot(panel: pd.DataFrame, cfg: Dict[str, Any], model: Dict[str, Any]) -> pd.DataFrame:
    """Scores of every member on the latest signal date; no forward-return columns are carried."""
    last = panel["date"].max()
    week = panel.loc[panel["date"] == last].copy()
    week["v8_ridge"] = vm.predict(week, cfg, model)
    ind_col = cfg.get("industry_col", "industry")
    has_ind = ind_col in week.columns
    keep = list(dict.fromkeys(["date", "ticker"] + [c for c in ("industry", ind_col) if c in week.columns]))
    out = week[keep].copy()
    out["stock_id"] = out["ticker"].str.split(".").str[0]
    for name in ("v8_zero", "v8_ridge", "pred_baseline_v6_prior"):
        out[name] = week[name]
    for arm, parts in ARM_SCORES.items():
        if arm not in cfg["arms"]:
            continue
        raw = vc.blend_ranks(week, parts)
        week[f"{arm}_raw"] = raw
        neut = vc.industry_demean(week, f"{arm}_raw", industry_col=ind_col) if has_ind else raw
        out[f"{arm}_score"] = neut
        out[f"{arm}_pct"] = neut.rank(pct=True)
    return out.reset_index(drop=True)


def watchlist_report(snap: pd.DataFrame, stock_ids: Sequence[str], arm: str, sell_pct: float) -> pd.DataFrame:
    by_id = snap.set_index("stock_id")
    rows: List[dict] = []
    for sid in stock_ids:
        if sid not in by_id.index:
            rows.append({"stock_id": sid, "pct": float("nan"), "status": "not in universe"})
            continue
        pct = float(by_id.at[sid, f"{arm}_pct"])
        rows.append({"stock_id": sid, "pct": round(pct, 3), "status": "sell zone" if pct < sell_pct else "hold zone"})
    return pd.DataFrame(rows)


def load_outcome(config_path: Path) -> Dict[str, Any]:
    """Recorded result of the pre-registered historical OOS (`v8_outcome.json` next to the config)."""
    path = Path(config_path).parent / "v8_outcome.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record_week(panel: pd.DataFrame, cfg: Dict[str, Any], model: Dict[str, Any], log_dir: Path,
                commit: str, dry_run: bool, outcome: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if cfg.get("status") != "FROZEN" and not dry_run:
        raise PermissionError("Config is not FROZEN; only --dry-run is allowed before the protocol is approved")
    snap = score_snapshot(panel, cfg, model)
    signal_date = pd.Timestamp(snap["date"].iloc[0])
    rec = {
        "signal_date": str(signal_date.date()),
        "status": (outcome or {}).get("paper_status", "PROTOCOL"),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_version": cfg.get("version", ""),
        "config_sha256": vc.config_hash(cfg),
        "model_train_last_date": model.get("train_last_date", ""),
        "commit": commit,
        "n_members": len(snap),
        "snapshot_file": f"snapshots/{signal_date.strftime('%Y%m%d')}.parquet",
        "dry_run": dry_run,
        "snapshot": snap,
    }
    if dry_run:
        return rec
    path = log_dir / rec["snapshot_file"]
    if path.exists():
        raise FileExistsError(f"Week {rec['signal_date']} already recorded; snapshots are append-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(path, index=False)
    manifest = log_dir / "manifest.csv"
    row = pd.DataFrame([{k: v for k, v in rec.items() if k != "snapshot"}])
    row.to_csv(manifest, mode="a", header=not manifest.exists(), index=False)
    return rec


def _print_summary(rec: Dict[str, Any], watch: Sequence[str], sell_pct: float) -> None:
    snap = rec["snapshot"]
    print(f"STATUS: {rec['status']}")
    print(f"Signal date {rec['signal_date']}  config {rec['config_sha256'][:12]}  members {rec['n_members']}"
          f"  {'DRY RUN (not recorded)' if rec['dry_run'] else 'recorded'}")
    for arm in ARM_SCORES:
        col = f"{arm}_pct"
        if col not in snap.columns:
            continue
        top = snap.nlargest(TOP_N_REPORT, col)[["stock_id", "industry", col]] if "industry" in snap else snap.nlargest(TOP_N_REPORT, col)
        print(f"\n{arm} top {TOP_N_REPORT}:\n{top.to_string(index=False)}")
        if watch:
            print(f"{arm} watchlist:\n{watchlist_report(snap, watch, arm, sell_pct).to_string(index=False)}")
    if rec["status"] != "PROTOCOL":
        print(f"\n{rec['status']}: comparison record only; not a trading signal, no real money.")
    print("\nModel signals only. No orders are placed; buy/sell decisions are the user's.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball v8 weekly paper-trading recorder (no orders)")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--watch", default="", help="Comma-separated stock ids to report, e.g. 2915,2881")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = vc.load_config(args.config)
    panel = pd.read_parquet(args.panel)
    rec = record_week(panel, cfg, vm.load_model(args.model), args.log_dir, _git_commit(), args.dry_run,
                      outcome=load_outcome(args.config))
    sell_pct = float(cfg.get("portfolios", {}).get("user_rules", {}).get("sell_pct", 0.60))
    _print_summary(rec, [s for s in args.watch.split(",") if s], sell_pct)


if __name__ == "__main__":
    main()
