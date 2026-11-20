"""Comprehensive TDD Tests for Snowball v8 Upgrades across 4 Pillars.

1. Data & Factors: Liquidity floor (`min_adv_usd`) + Sloan (1996) Accruals Quality (`compute_accruals_quality`)
2. Execution & Fill Feedback: Perold (1988) Implementation Shortfall (`reconcile_fills`) + Buffer-Band Sell Planner
3. Asset Allocation & Cash Management: Cash yield on vol-targeted idle cash + Core-Satellite allocator
4. MLOps & Model Decay Monitoring: Rolling IC / Brier / Industry concentration circuit breaker + Notebook A check
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mlops_monitor as mlops
import portfolio_sim as psim
import reconcile_fills as rf


def test_liquidity_filter_and_accruals_quality_sloan_1996():
    # Pillar 1A: Liquidity floor excludes illiquid micro-caps even if they have top score
    grp = pd.DataFrame([
        {"ticker": "ILLIQ.TW", "industry": "semi", "pred_m": 9.9, "adv_usd_20d": 100_000.0},
        {"ticker": "LIQ1.TW",  "industry": "semi", "pred_m": 2.0, "adv_usd_20d": 5_000_000.0},
        {"ticker": "LIQ2.TW",  "industry": "banks", "pred_m": 1.8, "adv_usd_20d": 4_000_000.0},
        {"ticker": "LIQ3.TW",  "industry": "tech", "pred_m": 1.5, "adv_usd_20d": 3_000_000.0},
        {"ticker": "LIQ4.TW",  "industry": "auto", "pred_m": 1.2, "adv_usd_20d": 2_000_000.0},
    ])
    selected = psim._select_cohort_target(grp, "pred_m", top_k=4, min_adv_usd=1_000_000.0)
    assert "ILLIQ.TW" not in selected, "Illiquid stock below min_adv_usd must be excluded"
    assert len(selected) == 4

    # Pillar 1B: Sloan (1996) Accruals Quality = -(NetIncome - OCF) / TotalAssets
    # High OCF vs NetIncome ("Blue Money") must score higher than low-cash accounting profit ("Red/Purple Money")
    fin = pd.DataFrame([
        {"ticker": "BLUE.TW", "net_income": 100.0, "ocf": 140.0, "total_assets": 1000.0},
        {"ticker": "RED.TW",  "net_income": 100.0, "ocf": -20.0, "total_assets": 1000.0},
    ])
    out = mlops.compute_accruals_quality(fin)
    q_blue = out.loc[out["ticker"] == "BLUE.TW", "accruals_quality"].iloc[0]
    q_red = out.loc[out["ticker"] == "RED.TW", "accruals_quality"].iloc[0]
    assert q_blue > 0.0 and q_red < 0.0 and q_blue > q_red


def test_cash_yield_boosts_vol_targeted_portfolio_and_core_satellite_allocator():
    # Pillar 3A: Idle cash under vol_target_annual earns cash_yield_annual instead of 0%
    dates = pd.date_range("2024-01-05", periods=28, freq="W-FRI")
    rows = []
    for i, d in enumerate(dates):
        shock = 0.05 * np.cos(i * 0.9)
        for j in range(12):
            rows.append({
                "date": d, "ticker": f"T{j}.TW", "market": "TW", "industry": f"ind{j % 4}",
                "pred_m": float(12 - j), "fwd_ret_1w_open": shock + 0.002,
            })
    df = pd.DataFrame(rows)
    no_yield = psim.simulate_topk_twr(df, "pred_m", top_k=4, rebalance_every=4, market="TW", vol_target_annual=0.12, cash_yield_annual=0.0)
    with_yield = psim.simulate_topk_twr(df, "pred_m", top_k=4, rebalance_every=4, market="TW", vol_target_annual=0.12, cash_yield_annual=0.04)
    assert with_yield["Avg_Exposure"] < 0.95
    assert with_yield["TWR_CAGR"] > no_yield["TWR_CAGR"], "Idle cash yield must increase net CAGR when de-risked"
    assert with_yield["TWR_Sharpe"] > no_yield["TWR_Sharpe"]

    # Pillar 3B: Core-Satellite allocator sums to 1.0 and increases defensive cash under high macro stress
    normal_alloc = mlops.allocate_core_satellite(macro_stress=0.0, realized_tw_vol=0.14)
    stress_alloc = mlops.allocate_core_satellite(macro_stress=2.0, realized_tw_vol=0.28)
    assert abs(sum(normal_alloc.values()) - 1.0) < 1e-6
    assert abs(sum(stress_alloc.values()) - 1.0) < 1e-6
    assert stress_alloc["cash_tbill"] > normal_alloc["cash_tbill"]
    assert stress_alloc["tw_alpha_satellite"] < normal_alloc["tw_alpha_satellite"]


def test_fill_reconciliation_measures_slippage_and_buffer_sell_plan(tmp_path: Path):
    # Pillar 2A: Implementation Shortfall (Perold 1988) in bps
    fills = pd.DataFrame([
        {"trade_date": "2026-11-20", "ticker": "2330.TW", "side": "buy", "ref_price": 1000.0, "fill_price": 1002.0, "qty": 100},
        {"trade_date": "2026-11-20", "ticker": "2412.TW", "side": "sell", "ref_price": 120.0, "fill_price": 119.4, "qty": 1000},
    ])
    log_csv = tmp_path / "fill_slippage_log.csv"
    summary = rf.reconcile_fills(fills, log_path=log_csv)
    assert log_csv.exists()
    # Buy paid +20 bps above ref; sell received -50 bps below ref -> mean shortfall = 35 bps
    assert abs(summary["mean_shortfall_bps"] - 35.0) < 1e-6
    assert summary["n_fills"] == 2

    # Pillar 2B: Buffer-Band Sell Planner ONLY sells held names that drop outside hold_top_pct
    ranked = pd.DataFrame([
        {"ticker": f"{2300 + i}.TW", "score": float(100 - i), "industry": f"ind{i % 5}"}
        for i in range(50)
    ])
    # 2305.TW is rank 6 (top 12% -> KEEP); 2335.TW is rank 36 (top 72% > 40% -> SELL)
    sells = rf.build_buffer_sell_plan(ranked, held_tickers=["2305.TW", "2335.TW"], hold_top_pct=0.40)
    assert sells == ["2335.TW"], f"Expected only out-of-band 2335.TW to be sold, got {sells}"


def test_mlops_model_decay_monitor_triggers_circuit_breaker():
    dates = pd.date_range("2025-01-03", periods=30, freq="W-FRI")
    healthy_ic = pd.Series(0.05, index=dates)
    decay_ic = pd.Series(-0.02, index=dates)
    ok_sig = pd.DataFrame([{"ticker": f"T{i}.TW", "industry": f"ind{i % 5}"} for i in range(20)])
    bad_sig = pd.DataFrame([{"ticker": f"T{i}.TW", "industry": "semi" if i < 10 else "banks"} for i in range(20)])

    h_rep = mlops.evaluate_model_health(healthy_ic, ok_sig, macro_stress=0.2)
    d_rep = mlops.evaluate_model_health(decay_ic, bad_sig, macro_stress=1.8)
    assert h_rep["status"] == "HEALTHY"
    assert d_rep["status"] == "DECAY_HALT"
    assert d_rep["max_industry_weight"] == 0.50
    assert d_rep["recommended_exposure_scale"] <= 0.50


def test_notebook_a_data_engine_valid_ipynb():
    repo_root = Path(__file__).resolve().parents[4]
    nb_a = repo_root / "notebooks" / "snowball_notebook_a_data_engine.ipynb"
    assert nb_a.exists(), f"Notebook A missing at {nb_a}"
    nb = json.loads(nb_a.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    src = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    assert "fetch_mops_monthly_revenue" in src
    assert "add_52w_high_proximity" in src
