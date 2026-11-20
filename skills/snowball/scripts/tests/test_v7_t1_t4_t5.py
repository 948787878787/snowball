"""TDD Tests for Snowball v7 Follow-Up Tasks T1 (Repro), T4 (Holdout Label Leak), and T5 (Excess Metrics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import stage5_validation as s5
import train_walkforward as twf


def test_t4_holdout_cutoff_masks_unsettled_forward_labels():
    dates = pd.date_range("2025-07-04", "2025-12-19", freq="W-FRI")
    rows = []
    for d in dates:
        for sym in ("2330.TW", "2412.TW"):
            rows.append({
                "date": d,
                "ticker": sym,
                "market": "TW",
                "fwd_ret_1w_open": 0.01,
                "fwd_ret_4w": 0.04,
                "fwd_excess_4w": 0.02,
                "fwd_vol_adj_4w": 0.5,
                "target_rank_4w": 0.8,
                "target_rank_mkt_4w": 0.8,
                "fwd_ret_12w": 0.12,
                "target_rank_12w": 0.7,
            })
    df = pd.DataFrame(rows)
    cutoff = pd.Timestamp("2025-11-20")
    masked = twf.mask_unsettled_holdout_labels(df, holdout_cutoff=cutoff)

    # All dates > cutoff must be removed
    assert (masked["date"] <= cutoff).all()

    # 1w next-day-open exit is ~2 weeks after signal Friday -> dates > cutoff - 2w must have NaN fwd_ret_1w_open
    late_1w = masked["date"] > (cutoff - pd.Timedelta(weeks=2))
    early_1w = masked["date"] <= (cutoff - pd.Timedelta(weeks=2))
    assert masked.loc[late_1w, "fwd_ret_1w_open"].isna().all()
    assert (masked.loc[early_1w, "fwd_ret_1w_open"] == 0.01).all()

    # 4w exit is (4+1)=5 weeks after signal date -> dates > cutoff - 5w must have NaN 4w labels
    late_4w = masked["date"] > (cutoff - pd.Timedelta(weeks=5))
    early_4w = masked["date"] <= (cutoff - pd.Timedelta(weeks=5))
    for col in ("fwd_ret_4w", "fwd_excess_4w", "fwd_vol_adj_4w", "target_rank_4w", "target_rank_mkt_4w"):
        assert masked.loc[late_4w, col].isna().all(), f"Leaked 4w label in {col}!"
        assert masked.loc[early_4w, col].notna().all(), f"Over-masked settled 4w label in {col}!"

    # 12w exit is (12+1)=13 weeks after signal date -> dates > cutoff - 13w must have NaN 12w labels
    late_12w = masked["date"] > (cutoff - pd.Timedelta(weeks=13))
    early_12w = masked["date"] <= (cutoff - pd.Timedelta(weeks=13))
    for col in ("fwd_ret_12w", "target_rank_12w"):
        assert masked.loc[late_12w, col].isna().all(), f"Leaked 12w label in {col}!"
        assert masked.loc[early_12w, col].notna().all(), f"Over-masked settled 12w label in {col}!"


def test_t5_compute_excess_metrics_zero_and_positive_excess():
    dates = pd.date_range("2022-01-07", periods=156, freq="W-FRI")
    rng = np.random.default_rng(123)
    bench = pd.Series(rng.normal(0.002, 0.02, len(dates)), index=dates)

    # Identical series -> zero excess, zero t-stat, zero IR, zero excess MDD
    zero_m = s5.compute_excess_metrics(bench, bench, lags=4)
    assert abs(zero_m["Net_Excess_CAGR"]) < 1e-12
    assert abs(zero_m["Net_Excess_Ann"]) < 1e-12
    assert abs(zero_m["Excess_t_NW"]) < 1e-12
    assert abs(zero_m["Net_IR"]) < 1e-12
    assert abs(zero_m["Excess_MDD"]) < 1e-12

    # Consistent positive excess -> positive excess, t_NW > 2.0, positive IR
    port_pos = bench + 0.003 + pd.Series(rng.normal(0.0, 0.003, len(dates)), index=dates)
    pos_m = s5.compute_excess_metrics(port_pos, bench, lags=4)
    assert pos_m["Net_Excess_Ann"] > 0.10
    assert pos_m["Excess_t_NW"] > 2.0
    assert pos_m["Net_IR"] > 1.0

    # New strict v7 pass gate: requires BOTH Excess_t_NW >= 2.0 AND conservative DSR >= 0.95
    assert s5.passes_v7_relative_gate(excess_t_nw=2.5, dsr_conservative=0.96) is True
    assert s5.passes_v7_relative_gate(excess_t_nw=2.5, dsr_conservative=0.80) is False
    assert s5.passes_v7_relative_gate(excess_t_nw=1.5, dsr_conservative=0.99) is False


def test_t1_run_stage5_holdout_script_exists_in_repo():
    script_path = Path(__file__).resolve().parents[1] / "run_stage5_holdout.py"
    assert script_path.exists(), f"Reproducible holdout script missing from repo: {script_path}"
    import run_stage5_holdout as r5
    assert hasattr(r5, "evaluate_oos_periods")
    assert hasattr(r5, "audit_existing_experiments")
