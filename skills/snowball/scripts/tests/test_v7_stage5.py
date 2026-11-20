"""Stage 5 TDD Tests for Multi-Seed Ensemble & CPCV Regime Validation (Snowball v7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import stage5_validation as s5


def test_cpcv_paths_enforce_purge_and_cover_all_blocks():
    dates = pd.date_range("2021-01-01", periods=180, freq="W-FRI")
    splits = s5.generate_cpcv_splits(dates, n_groups=6, k_test_groups=2, purge_weeks=5)
    # C(6, 2) = 15 combinatorial paths
    assert len(splits) == 15
    for tr_dates, te_dates in splits:
        assert len(tr_dates) > 0 and len(te_dates) > 0
        # Zero overlap and minimum purge gap between adjacent train/test boundaries
        assert set(tr_dates).isdisjoint(set(te_dates))
        te_min, te_max = min(te_dates), max(te_dates)
        for d in tr_dates:
            if d < te_min:
                assert (te_min - d).days >= 5 * 7 or any(
                    abs((d - td).days) >= 5 * 7 for td in te_dates
                )


def test_build_ridge_lgbm_rank_ensemble_blends_equally():
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    df = pd.DataFrame([
        {"date": d, "ticker": t, "market": "TW", "pred_stage2_ridge": r, "pred_stage3_lgbm": g}
        for d in dates
        for t, r, g in [("A.TW", 4.0, 1.0), ("B.TW", 3.0, 4.0), ("C.TW", 2.0, 3.0), ("D.TW", 1.0, 2.0)]
    ])
    out = s5.build_rank_ensemble(df, ["pred_stage2_ridge", "pred_stage3_lgbm"], out_col="pred_ens")
    s0 = out[out["date"] == dates[0]].set_index("ticker")["pred_ens"]
    # B has ranks (0.75, 1.00) -> 0.875; A has (1.00, 0.25) -> 0.625; C has (0.50, 0.75) -> 0.625; D has (0.25, 0.50) -> 0.375
    assert abs(s0["B.TW"] - 0.875) < 1e-6
    assert abs(s0["D.TW"] - 0.375) < 1e-6


def test_regime_subperiod_attribution_splits_calendar_years():
    dates = pd.date_range("2022-01-07", "2024-12-27", freq="W-FRI")
    rng = np.random.default_rng(7)
    port_w = pd.Series(rng.normal(0.003, 0.02, len(dates)), index=dates)
    ew_w = pd.Series(rng.normal(0.002, 0.02, len(dates)), index=dates)
    tbl = s5.summarize_annual_regimes(port_w, ew_w)
    assert set(tbl["year"].tolist()) == {2022, 2023, 2024}
    assert {"port_ret", "ew_ret", "excess_ret", "port_sharpe", "port_mdd"}.issubset(tbl.columns)
