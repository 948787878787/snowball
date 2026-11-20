"""Reproducible Stage 5 Holdout & Relative-to-EW Audit Script (Snowball v7 T1 & T5).

Guarantees full reproducibility from a clean checkout:
- Reproduces `stage5_final_summary.json` for Pre-Holdout (<= 2025-11-20), Used Holdout (> 2025-11-20),
  and Full OOS (2021-10 -> 2026-11) using the locked v7 ensemble (0.5*Ridge + 0.5*3-Seed LGBM across
  0.5*4w + 0.5*12w) and Stage 3 portfolio controls (TW Top-20, IndCap <= 25%, Neutralized, Buffer 10x).
- Computes T5 relative-to-TW-EW metrics (Net Excess CAGR, Net Excess Ann, Excess t_NW, Net IR,
  Excess MDD) and conservative Deflated Sharpe Ratio (N_cum + 20) for every experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate as ev
import portfolio_sim as psim
import stage5_validation as s5
import train_walkforward as twf
import tw_monthly_rev as tmr

HOLDOUT_CUTOFF = pd.Timestamp("2025-11-20")
STAGE3_LOCKED_CFG = dict(max_industry_weight=0.25, neutralize_scores=True, buffer_mult=10.0)


def _run_3seed_walkforward(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Expanding purged walk-forward (purge = horizon + 1) with Ridge + 3-Seed Monotone LightGBM."""
    runs = [
        twf.run_purged_walkforward(
            panel=panel, horizon=horizon, min_train_weeks=104, refit_step_weeks=8,
            ridge_alpha=500.0, enable_lgbm=True, use_gpu=False, top_k=20, seed=sd,
        )
        for sd in (42, 137, 2026)
    ]
    oos = runs[0]["oos_predictions"].copy()
    oos["pred_lgbm3"] = np.mean([r["oos_predictions"]["pred_stage3_lgbm"].to_numpy() for r in runs], axis=0)
    return oos


def build_v7_final_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    """Train the locked v7 4w+12w Ridge + 3-Seed LGBM ensemble on `panel`."""
    oos_4w = _run_3seed_walkforward(panel, horizon=4)
    oos_12w = _run_3seed_walkforward(panel, horizon=12)
    b_r = tmr.blend_horizon_predictions(oos_4w, oos_12w, score_col="pred_stage2_ridge", w_4w=0.5, out_col="r_blend")
    b_l = tmr.blend_horizon_predictions(oos_4w, oos_12w, score_col="pred_lgbm3", w_4w=0.5, out_col="l_blend")
    out = b_r.copy()
    out["l_blend"] = b_l["l_blend"].to_numpy()
    out["pred_v7_final"] = 0.5 * out["r_blend"] + 0.5 * out["l_blend"]
    return out


def _eval_one_slice(
    sub: pd.DataFrame, score_col: str, ew_w: pd.Series, t_col: str, cfg: dict, n_trials_cons: int,
) -> Dict[str, float]:
    ic_s = twf._compute_daily_ic_series(sub, score_col, t_col, market_filter="TW")
    ic_m = float(ic_s.mean()) if len(ic_s) else np.nan
    tnw_m = float(ev.newey_west_t(ic_s, lags=4)) if len(ic_s) >= 6 else np.nan
    sb = psim.simulate_topk_twr(sub, score_col, top_k=20, rebalance_every=4, market="TW", tw_odd_lot=False, **cfg)
    so = psim.simulate_topk_twr(sub, score_col, top_k=20, rebalance_every=4, market="TW", tw_odd_lot=True, **cfg)
    port_w = s5.compute_portfolio_weekly_returns(sub, score_col, top_k=20, rebalance_every=4, market="TW", tw_odd_lot=False, **cfg)
    ew_sub = ew_w.loc[ew_w.index.isin(port_w.index)]
    ex = s5.compute_excess_metrics(port_w, ew_sub, lags=4)
    n_w = int(len(port_w))
    dsr_cons = ev.deflated_sharpe_ratio(sb["TWR_Sharpe"], n_trials=n_trials_cons, n_obs=n_w)
    return {
        "n_weeks": n_w, "TW_IC": ic_m, "TW_tNW": tnw_m,
        "Board_CAGR": sb["TWR_CAGR"], "Board_Sharpe": sb["TWR_Sharpe"], "MDD": sb["TWR_MDD"],
        "Odd_CAGR": so["TWR_CAGR"], "Odd_Sharpe": so["TWR_Sharpe"], "Turn": sb["Avg_Rebalance_Turnover"],
        **ex, "DSR_cons": dsr_cons,
        "Pass_T5_Gate": s5.passes_v7_relative_gate(ex["Excess_t_NW"], dsr_cons),
    }


def evaluate_oos_periods(
    full_oos: pd.DataFrame, *, score_col: str = "pred_v7_final", n_trials_cons: int = 65,
) -> Dict[str, Dict[str, float]]:
    """Evaluate Pre-Holdout, Used Holdout (> 2025-11-20), and Full OOS with absolute & T5 relative metrics."""
    df = full_oos.copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    ew_w = df.loc[df["market"] == "TW"].groupby("date")["fwd_ret_1w_open"].mean().dropna()
    t_col = "target_rank_4w" if "target_rank_4w" in df.columns else "target_rank_mkt_4w"
    slices = [
        ("Pre-Holdout (<= 2025-11-20)", df["date"] <= HOLDOUT_CUTOFF),
        ("USED 52W HOLDOUT (> 2025-11-20)", df["date"] > HOLDOUT_CUTOFF),
        ("Full OOS (2021-10 -> 2026-11)", pd.Series(True, index=df.index)),
    ]
    return {
        label: _eval_one_slice(df.loc[mask].copy(), score_col, ew_w, t_col, STAGE3_LOCKED_CFG, n_trials_cons)
        for label, mask in slices
    }


def audit_existing_experiments(base_dir: Path, n_trials_cons: int = 65) -> pd.DataFrame:
    """Re-score existing Stage 1, Stage 3, Stage 4, and Stage 5 TW artifacts using T5 relative-to-EW metrics."""
    specs = [
        ("Stage1_Combined_Base", base_dir / "snowball_artifacts_v7_stage1/oos_predictions.parquet",
         "pred_baseline_v6_prior", dict(max_industry_weight=None, neutralize_scores=False, buffer_mult=1.0), True),
        ("Stage1_Combined_ICIR", base_dir / "snowball_artifacts_v7_stage1/oos_predictions.parquet",
         "pred_stage1_icir_shrinkage", dict(max_industry_weight=None, neutralize_scores=False, buffer_mult=1.0), True),
        ("Stage3_TW_Prior", base_dir / "snowball_artifacts_tw_v7/oos_predictions.parquet",
         "pred_baseline_v6_prior", STAGE3_LOCKED_CFG, True),
        ("Stage3_TW_ICIR", base_dir / "snowball_artifacts_tw_v7/oos_predictions.parquet",
         "pred_stage1_icir_shrinkage", STAGE3_LOCKED_CFG, True),
        ("Stage3_TW_Ridge", base_dir / "snowball_artifacts_tw_v7/oos_predictions.parquet",
         "pred_stage2_ridge", STAGE3_LOCKED_CFG, True),
        ("Stage3_TW_LGBM", base_dir / "snowball_artifacts_tw_v7/oos_predictions.parquet",
         "pred_stage3_lgbm", STAGE3_LOCKED_CFG, True),
        ("Stage4_TW_Ridge_4w", base_dir / "snowball_artifacts_tw_v7_stage4/stage4_4w/oos_predictions.parquet",
         "pred_stage2_ridge", STAGE3_LOCKED_CFG, True),
        ("Stage4_TW_LGBM_4w", base_dir / "snowball_artifacts_tw_v7_stage4/stage4_4w/oos_predictions.parquet",
         "pred_stage3_lgbm", STAGE3_LOCKED_CFG, True),
        ("Stage5_Final_PreHoldout", base_dir / "snowball_artifacts_tw_v7_final/v7_final_oos_predictions.parquet",
         "pred_v7_final", STAGE3_LOCKED_CFG, True),
        ("Stage5_Final_USED_Holdout", base_dir / "snowball_artifacts_tw_v7_final/v7_final_oos_predictions.parquet",
         "pred_v7_final", STAGE3_LOCKED_CFG, False),
    ]
    rows: List[dict] = []
    for name, path, sc, cfg, pre_only in specs:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df = df.loc[df["market"] == "TW"].copy()
        df = df.loc[df["date"] <= HOLDOUT_CUTOFF] if pre_only else df.loc[df["date"] > HOLDOUT_CUTOFF]
        ew_w = df.groupby("date")["fwd_ret_1w_open"].mean().dropna()
        t_col = "target_rank_4w" if "target_rank_4w" in df.columns else "target_rank_mkt_4w"
        rec = _eval_one_slice(df, sc, ew_w, t_col, cfg, n_trials_cons)
        rows.append({"experiment": name, **rec})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Stage 5 Holdout & T5 Relative-to-EW Audit")
    parser.add_argument("--panel", type=Path, default=None, help="Stage 4 TW panel parquet (trains from scratch)")
    parser.add_argument("--oos-parquet", type=Path, default=None, help="Existing v7_final_oos_predictions.parquet")
    parser.add_argument("--audit-base-dir", type=Path, default=None, help="Base dir containing Stage 1..5 artifact folders")
    parser.add_argument("--trial-log", type=Path, default=Path("results/trial_log.csv"), help="Path to trial_log.csv")
    parser.add_argument("--append-t1-correction", action="store_true", help="Append T1 correction row to trial_log.csv")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save JSON summary")
    args = parser.parse_args()

    n_logged = len(pd.read_csv(args.trial_log)) if args.trial_log.exists() else 44
    n_cons = n_logged + 20

    if args.panel is not None:
        panel = pd.read_parquet(args.panel)
        panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
        full_oos = build_v7_final_predictions(panel)
    elif args.oos_parquet is not None:
        full_oos = pd.read_parquet(args.oos_parquet)
    else:
        full_oos = None

    if full_oos is not None:
        periods = evaluate_oos_periods(full_oos, n_trials_cons=n_cons)
        print(f"=== Reproducible Stage 5 Holdout Summary (Conservative N={n_cons}) ===")
        print(pd.DataFrame.from_dict(periods, orient="index").round(4).to_string())
        if args.append_t1_correction:
            h = periods["USED 52W HOLDOUT (> 2025-11-20)"]
            tid = psim.append_trial_log(
                args.trial_log,
                stage="Stage5_FINAL_HOLDOUT_52W_T1_CORRECTION",
                market="TW",
                model="v7_final_ridge_lgbm3_4w12w",
                change="T1 correction of trial #44 (fixed unsettled 2026-11-13 tail -30% bug in 7d3822b); HOLD-OUT MARKED USED",
                ic=h["TW_IC"], t_nw=h["TW_tNW"], cagr=h["Board_CAGR"], sharpe=h["Board_Sharpe"],
                mdd=h["MDD"], turnover=h["Turn"],
            )
            print(f"Appended T1 correction row trial_id={tid} to {args.trial_log}")
        if args.out_json is not None:
            args.out_json.write_text(json.dumps({"n_trials_cons": n_cons, "periods": periods}, indent=2, default=str), encoding="utf-8")

    if args.audit_base_dir is not None:
        tbl = audit_existing_experiments(args.audit_base_dir, n_trials_cons=n_cons)
        pd.set_option("display.width", 260)
        pd.set_option("display.max_columns", 20)
        print(f"\n=== T5 Relative-to-TW-EW Audit Across All Existing Experiments (Conservative N={n_cons}) ===")
        cols = ["experiment", "n_weeks", "Board_CAGR", "Board_Sharpe", "Net_Excess_CAGR", "Excess_t_NW", "Net_IR", "Excess_MDD", "DSR_cons", "Pass_T5_Gate"]
        print(tbl[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
