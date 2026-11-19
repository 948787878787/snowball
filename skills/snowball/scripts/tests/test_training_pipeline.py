"""Tests for the US+TW 2,000-stock Point-in-Time training panel builder & Purged Walk-Forward engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_training_panel as btp
import global_ranker as gr
import train_walkforward as twf


def test_combined_us_tw_panel_schema_and_next_open_targets():
    panel = btp.build_combined_panel(
        markets=("US", "TW"),
        horizons=(4, 12),
        offline_demo=True,
        demo_stocks_per_market=20,
        demo_days=780,
    )
    assert set(panel["market"].unique()) == {"US", "TW"}
    assert panel["ticker"].nunique() >= 35
    for col in (
        "date",
        "ticker",
        "market",
        "industry",
        "f_mom_qual",
        "f_ind_mom",
        "f_low_risk",
        "f_rev_1w",
        "macro_stress",
        "rate_inflation",
        "ix_stress_x_lowrisk",
        "fwd_ret_1w_open",
        "fwd_ret_4w",
        "fwd_excess_4w",
        "target_rank_4w",
        "target_rank_global_4w",
    ):
        assert col in panel.columns, f"Missing expected column: {col}"

    valid_ranks = panel["target_rank_4w"].dropna()
    assert len(valid_ranks) > 500
    assert valid_ranks.min() >= 0.0 and valid_ranks.max() <= 1.0


def test_pit_fundamentals_merge_has_zero_lookahead(tmp_path: Path):
    panel = btp.build_combined_panel(
        markets=("US", "TW"),
        horizons=(4,),
        offline_demo=True,
        demo_stocks_per_market=15,
        demo_days=650,
    )
    tickers = sorted(panel["ticker"].unique())
    mid_date = sorted(panel["date"].unique())[len(panel["date"].unique()) // 2]

    # Create PIT fundamentals available ONLY on/after mid_date
    pit_rows = [
        {"ticker": t, "available_date": pd.Timestamp(mid_date).date().isoformat(), "rule40_pct": float(20 + (i % 5) * 10)}
        for i, t in enumerate(tickers)
    ]
    pit_csv = tmp_path / "pit_fund.csv"
    pd.DataFrame(pit_rows).to_csv(pit_csv, index=False)

    merged = btp._merge_pit_fundamentals(panel, pit_csv)
    assert "pit_rule40_z" in merged.columns

    before = merged.loc[merged["date"] < mid_date, "pit_rule40_z"]
    after = merged.loc[merged["date"] >= mid_date, "pit_rule40_z"]
    assert before.isna().all(), "Future PIT filing leaked into earlier dates!"
    assert after.notna().any(), "PIT filing should be active on/after available_date"


def test_purged_walkforward_enforces_purge_gap_and_exports_artifacts(tmp_path: Path):
    panel = btp.build_combined_panel(
        markets=("US", "TW"),
        horizons=(4,),
        offline_demo=True,
        demo_stocks_per_market=25,
        demo_days=860,
    )
    horizon = 4
    res = twf.run_purged_walkforward(
        panel=panel,
        horizon=horizon,
        min_train_weeks=40,
        refit_step_weeks=8,
        ridge_alpha=300.0,
        enable_lgbm=False,
        top_k=10,
    )

    # 1. Verify strict purge gap on every refit
    assert len(res["purge_audit"]) >= 3
    for entry in res["purge_audit"]:
        assert entry["gap_days"] >= horizon * 7, f"Purge gap violated: {entry}"

    # 2. Verify all 3 core models evaluated with US & TW splits
    for m_name in ("baseline_v6_prior", "stage1_icir_shrinkage", "stage2_ridge"):
        m = res["metrics"][m_name]
        assert m["OOS_Weeks"] >= 20
        assert np.isfinite(m["Rank_IC"])
        assert np.isfinite(m["US_Rank_IC"])
        assert np.isfinite(m["TW_Rank_IC"])
        assert np.isfinite(m["TWR_Sharpe"])

    # 3. Verify portable artifact export and global_ranker consumption
    exported = twf.export_training_artifacts(res, tmp_path / "artifacts")
    w_path = Path(exported["v6_learned_weights"])
    assert w_path.exists()
    assert Path(exported["oos_summary"]).exists()
    assert Path(exported["weight_history"]).exists()
    assert Path(exported["oos_predictions"]).exists()

    loaded_w = gr.load_learned_weights(w_path)
    assert set(loaded_w.keys()) == set(gr.BASELINE_V6_WEIGHTS.keys())
    assert abs(sum(loaded_w.values()) - 1.0) < 1e-6

    # Feed loaded Kaggle weights directly into global_ranker.score_latest_global
    groups = {"tech": ("AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN")}
    close_d, open_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
    pf = gr.compute_global_price_factors(close_d, bench_d, groups, open_d=open_d)
    ranked = gr.score_latest_global(pf, fund_df, groups, model="v6", learned_weights=loaded_w)
    assert len(ranked) == 6
    assert "v6_score" in ranked.columns and ranked["v6_score"].notna().all()


def test_kaggle_notebook_is_valid_nbformat_v4():
    nb_path = Path(__file__).resolve().parents[4] / "notebooks" / "snowball_kaggle_train.ipynb"
    assert nb_path.exists(), f"Kaggle notebook missing: {nb_path}"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 6
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4
    joined_code = "\n".join("".join(c["source"]) for c in code_cells)
    joined_all = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "build_combined_panel" in joined_code
    assert "run_purged_walkforward" in joined_code
    assert "export_training_artifacts" in joined_code
    assert "v6_learned_weights.json" in joined_all
