"""Stage 3 TDD Tests for Taiwan Portfolio Construction Layer & Notebook B (Snowball v7).

Tests the 5 sequential Stage 3 portfolio construction controls in `portfolio_sim.py`:
1. Single-market isolation (`market="TW"`)
2. Industry concentration cap (`max_industry_weight=0.25`)
3. Cross-sectional Beta / volatility / industry neutralization (`neutralize_cross_section`)
4. Turnover buffer band (`buffer_mult=2.0` / keep held stocks while in top 40%)
5. Ex-ante trailing volatility targeting (`vol_target_annual=0.15`, zero look-ahead)
Plus structural validation of `notebooks/snowball_notebook_b_model_lab.ipynb`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio_sim as psim


def _make_tw_synthetic_oos(n_weeks: int = 24, n_stocks: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-05", periods=n_weeks, freq="W-FRI")
    industries = ["semiconductors", "shipping", "financials", "biotech", "plastics"]
    rng = np.random.default_rng(42)
    rows = []
    base_scores = np.linspace(2.0, -2.0, n_stocks)
    for w_idx, dt in enumerate(dates):
        mkt_shock = 0.045 * np.cos(w_idx * 0.9) if w_idx >= 4 else 0.01 * np.cos(w_idx)
        for s_idx in range(n_stocks):
            # First 35% of stocks belong to semiconductors so unconstrained top-k is >75% semi
            ind = "semiconductors" if s_idx < max(15, n_stocks // 3) else industries[(s_idx % (len(industries) - 1)) + 1]
            vol = 0.40 if s_idx % 2 == 0 else 0.15
            # Add small weekly jitter around base rank so top-10 and top-15 swap places without buffer
            jitter = 0.35 * np.sin(w_idx * 1.3 + s_idx)
            score = float(base_scores[s_idx] + jitter + 1.5 * vol)
            ret = float(mkt_shock + rng.normal(0.002, 0.015))
            rows.append({
                "date": dt,
                "ticker": f"{2300 + s_idx}.TW",
                "market": "TW",
                "industry": ind,
                "f_vol_60d": vol,
                "f_low_risk": -vol,
                "pred_m": score,
                "fwd_ret_1w_open": ret,
            })
    return pd.DataFrame(rows)


def test_industry_cap_limits_single_industry_concentration():
    df = _make_tw_synthetic_oos(n_weeks=4, n_stocks=120)
    grp = df[df["date"] == df["date"].min()]
    # Unconstrained top_k=12 picks mostly semiconductors
    unconstrained = psim._select_cohort_target(grp, "pred_m", top_k=12)
    semi_uncon = grp[grp["ticker"].isin(unconstrained)]["industry"].value_counts().get("semiconductors", 0)
    assert semi_uncon > 3

    # With max_industry_weight=0.25 on k=12, at most 3 stocks (25% of 12) can come from semiconductors
    capped = psim._select_cohort_target(grp, "pred_m", top_k=12, max_industry_weight=0.25)
    assert len(capped) == 12
    ind_counts = grp[grp["ticker"].isin(capped)]["industry"].value_counts()
    assert ind_counts.max() <= 3, f"Industry cap 25% violated on k=12: {ind_counts.to_dict()}"


def test_cross_sectional_neutralization_orthogonalizes_vol_and_industry():
    df = _make_tw_synthetic_oos(n_weeks=8, n_stocks=40)
    neut = psim.neutralize_score_panel(df, "pred_m", risk_cols=("f_vol_60d", "f_low_risk"), industry_col="industry")
    assert "pred_m_neut" in neut.columns
    for _, grp in neut.groupby("date"):
        corr_vol = np.corrcoef(grp["pred_m_neut"], grp["f_vol_60d"])[0, 1]
        assert abs(corr_vol) < 1e-6, f"Residual score still correlated with volatility: {corr_vol}"
        ind_means = grp.groupby("industry")["pred_m_neut"].mean().abs().max()
        assert ind_means < 1e-6, f"Residual score not industry-neutral: max mean={ind_means}"


def test_buffer_band_reduces_rebalance_turnover():
    df = _make_tw_synthetic_oos(n_weeks=24, n_stocks=40)
    no_buf = psim.simulate_topk_twr(df, "pred_m", top_k=10, rebalance_every=4, market="TW", buffer_mult=1.0)
    with_buf = psim.simulate_topk_twr(df, "pred_m", top_k=10, rebalance_every=4, market="TW", buffer_mult=2.0)
    assert with_buf["Avg_Rebalance_Turnover"] < no_buf["Avg_Rebalance_Turnover"] - 0.10, (
        f"Expected buffer band to reduce turnover significantly: no_buf={no_buf['Avg_Rebalance_Turnover']:.3f}, "
        f"with_buf={with_buf['Avg_Rebalance_Turnover']:.3f}"
    )


def test_ex_ante_vol_targeting_scales_down_volatility_and_mdd():
    df = _make_tw_synthetic_oos(n_weeks=28, n_stocks=40)
    unscaled = psim.simulate_topk_twr(df, "pred_m", top_k=10, rebalance_every=4, market="TW", vol_target_annual=None)
    targeted = psim.simulate_topk_twr(df, "pred_m", top_k=10, rebalance_every=4, market="TW", vol_target_annual=0.12)
    assert abs(targeted["TWR_MDD"]) < abs(unscaled["TWR_MDD"]), (
        f"Vol target should reduce MDD magnitude: targeted={targeted['TWR_MDD']}, unscaled={unscaled['TWR_MDD']}"
    )
    assert targeted.get("Realized_Annual_Vol", 1.0) < unscaled.get("Realized_Annual_Vol", 0.0)


def test_notebook_b_model_lab_exists_and_enforces_v7_principles():
    repo_root = Path(__file__).resolve().parents[4]
    nb_path = repo_root / "notebooks" / "snowball_notebook_b_model_lab.ipynb"
    assert nb_path.exists(), f"Notebook B missing at {nb_path}"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    full_src = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    assert "2025-11-20" in full_src, "Notebook B must enforce 52-week holdout cutoff 2025-11-20"
    assert "deflated_sharpe_ratio" in full_src, "Notebook B must compute Deflated Sharpe Ratio"
    assert "max_industry_weight" in full_src and "buffer_mult" in full_src, "Notebook B must run Stage 3 TW controls"
