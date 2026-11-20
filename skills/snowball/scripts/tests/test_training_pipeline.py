"""Tests for the US+TW 2,000-stock Point-in-Time training panel builder & Purged Walk-Forward engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_training_panel as btp
import fetch_pit_fundamentals as fpf
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


def test_constrained_ridge_forbids_high_volatility_chasing():
    rng = np.random.default_rng(7)
    n = 400
    f_mom = rng.normal(0, 1, n)
    f_vol = rng.normal(0, 1, n)
    f_low = -f_vol + rng.normal(0, 0.2, n)
    # Artificially reward high volatility (mimics bull-market survivorship trap)
    y = 0.5 + 0.15 * f_vol + 0.05 * f_mom + rng.normal(0, 0.02, n)
    feats = ["f_mom_qual", "f_vol_60d", "f_low_risk"]
    X = np.column_stack([f_mom, f_vol, f_low])
    _, coef = twf._fit_ridge(X, y, l2_alpha=10.0, feature_names=feats)
    c_map = dict(zip(feats, coef))
    assert c_map["f_mom_qual"] >= 0.0
    assert c_map["f_vol_60d"] <= 1e-12, f"Constrained Ridge must not chase high volatility: {c_map}"
    assert c_map["f_low_risk"] >= -1e-12, f"Constrained Ridge must not short low_risk: {c_map}"


def test_sec_xbrl_parser_and_pit_merge_handles_datetime_resolution():
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "start": "2023-01-01", "end": "2023-03-31", "filed": "2023-05-01", "val": 1000.0},
                            {"form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 1300.0},
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 390.0},
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 65.0},
                        ]
                    }
                },
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {"form": "10-K", "start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-15", "val": 5.0},
                        ]
                    }
                },
            }
        }
    }
    pit_df = fpf.parse_sec_company_facts("AAPL", payload, min_year=2023)
    assert len(pit_df) == 2
    latest = pit_df.iloc[-1]
    assert latest["available_date"] == "2024-05-02"  # filed + 1 day
    assert abs(latest["rev_yoy"] - 0.30) < 1e-6
    assert abs(latest["fcf_margin"] - 0.25) < 1e-6
    assert abs(latest["rule40_pct"] - 55.0) < 1e-6

    # Merge into panel with datetime64[ns] dates (tests Pandas 3.0 ns vs us resolution safety)
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-04-26", "2024-05-03", "2024-05-03", "2024-05-03", "2024-05-03"]).astype("datetime64[ns]"),
        "ticker": ["AAPL", "AAPL", "MSFT", "NVDA", "GOOGL"],
        "market": ["US"] * 5,
        "industry": ["tech"] * 5,
        "close_usd": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    # Add peers on 2024-05-02 so cross-sectional z-score has std > 0
    peers = pd.DataFrame([
        {"ticker": "MSFT", "available_date": "2024-05-02", "rule40_pct": 40.0, "rev_yoy": 0.15, "fcf_margin": 0.25, "ttm_eps": 4.0, "ep_yield": np.nan, "roe": 0.3, "gross_margin_chg": 0.01},
        {"ticker": "NVDA", "available_date": "2024-05-02", "rule40_pct": 80.0, "rev_yoy": 0.50, "fcf_margin": 0.30, "ttm_eps": 3.0, "ep_yield": np.nan, "roe": 0.5, "gross_margin_chg": 0.05},
        {"ticker": "GOOGL", "available_date": "2024-05-02", "rule40_pct": 35.0, "rev_yoy": 0.10, "fcf_margin": 0.25, "ttm_eps": 6.0, "ep_yield": np.nan, "roe": 0.25, "gross_margin_chg": 0.0},
    ])
    full_pit = pd.concat([pit_df, peers], ignore_index=True)
    merged = fpf.merge_pit_into_panel(panel, full_pit)
    # 2024-04-26 is BEFORE 2024-05-02 -> AAPL rule40 must be NaN (zero look-ahead!)
    before_val = merged.loc[(merged["ticker"] == "AAPL") & (merged["date"] == pd.Timestamp("2024-04-26")), "pit_rule40_z"].iloc[0]
    after_val = merged.loc[(merged["ticker"] == "AAPL") & (merged["date"] == pd.Timestamp("2024-05-03")), "pit_rule40_z"].iloc[0]
    assert pd.isna(before_val)
    assert np.isfinite(after_val)


def test_p0_6_stale_pit_filings_expire_after_tolerance():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-05-10", "2024-05-10", "2024-05-10", "2024-05-10", "2025-03-07", "2025-03-07", "2025-03-07", "2025-03-07"]).astype("datetime64[ns]"),
        "ticker": ["A", "B", "C", "D", "A", "B", "C", "D"],
        "market": ["US"] * 8,
        "industry": ["tech"] * 8,
        "close_usd": [100.0] * 8,
        "raw_close_usd": [100.0] * 8,
    })
    # Stock A only filed in May 2024 (stale >180d by March 2025); B,C,D filed fresh in Feb 2025
    pit = pd.DataFrame([
        {"ticker": "A", "available_date": "2024-05-01", "period_end": "2024-03-31", "rule40_pct": 60.0},
        {"ticker": "B", "available_date": "2024-05-01", "period_end": "2024-03-31", "rule40_pct": 40.0},
        {"ticker": "C", "available_date": "2024-05-01", "period_end": "2024-03-31", "rule40_pct": 30.0},
        {"ticker": "D", "available_date": "2024-05-01", "period_end": "2024-03-31", "rule40_pct": 20.0},
        {"ticker": "B", "available_date": "2025-02-15", "period_end": "2024-12-31", "rule40_pct": 42.0},
        {"ticker": "C", "available_date": "2025-02-15", "period_end": "2024-12-31", "rule40_pct": 31.0},
        {"ticker": "D", "available_date": "2025-02-15", "period_end": "2024-12-31", "rule40_pct": 21.0},
    ])
    merged = fpf.merge_pit_into_panel(panel, pit, max_staleness_days=180)
    a_fresh = merged.loc[(merged["ticker"] == "A") & (merged["date"] == pd.Timestamp("2024-05-10")), "pit_rule40_z"].iloc[0]
    a_stale = merged.loc[(merged["ticker"] == "A") & (merged["date"] == pd.Timestamp("2025-03-07")), "pit_rule40_z"].iloc[0]
    assert np.isfinite(a_fresh)
    assert pd.isna(a_stale), "Stale filing >180d must expire to NaN!"


def test_p0_6_intra_ttm_split_normalizes_eps():
    # 3 pre-split quarters ($5 EPS, 100M shares) + 1 post-10:1-split quarter ($0.50 EPS, 1,000M shares)
    # True post-split TTM EPS = 0.5 + 0.5 + 0.5 + 0.5 = $2.00 (NOT 5+5+5+0.5 = $15.50!)
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "start": "2023-07-01", "end": "2023-09-30", "filed": "2023-11-01", "val": 1000.0},
                            {"form": "10-Q", "start": "2023-10-01", "end": "2023-12-31", "filed": "2024-02-01", "val": 1000.0},
                            {"form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 1000.0},
                            {"form": "10-Q", "start": "2024-04-01", "end": "2024-06-30", "filed": "2024-08-01", "val": 1000.0},
                        ]
                    }
                },
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {"form": "10-Q", "start": "2023-07-01", "end": "2023-09-30", "filed": "2023-11-01", "val": 5.0},
                            {"form": "10-Q", "start": "2023-10-01", "end": "2023-12-31", "filed": "2024-02-01", "val": 5.0},
                            {"form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 5.0},
                            {"form": "10-Q", "start": "2024-04-01", "end": "2024-06-30", "filed": "2024-08-01", "val": 0.5},
                        ]
                    }
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"form": "10-Q", "start": "2023-07-01", "end": "2023-09-30", "filed": "2023-11-01", "val": 1e8},
                            {"form": "10-Q", "start": "2023-10-01", "end": "2023-12-31", "filed": "2024-02-01", "val": 1e8},
                            {"form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 1e8},
                            {"form": "10-Q", "start": "2024-04-01", "end": "2024-06-30", "filed": "2024-08-01", "val": 1e9},
                        ]
                    }
                },
            }
        }
    }
    out = fpf.parse_sec_company_facts("NVDA", payload, min_year=2023)
    latest_eps = out.iloc[-1]["ttm_eps"]
    assert abs(latest_eps - 2.0) < 1e-6, f"Expected split-normalized TTM EPS=2.0, got {latest_eps}"


def test_p0_6_split_immune_ep_yield_guard_and_splits_df():
    # NVDA pre-split in May 2024: nominal raw_close=$1000, unadjusted TTM EPS=$20 (true E/P=2%).
    # After 10-for-1 split in June 2024, Yahoo auto_adjust close_usd in May 2024 is $100.
    # Peer stocks have true E/P = 4%, 5%, 6% (higher than NVDA's 2%).
    panel_adj_only = pd.DataFrame({
        "date": pd.to_datetime(["2024-05-10"] * 4 + ["2024-07-12"]).astype("datetime64[ns]"),
        "ticker": ["NVDA", "P1", "P2", "P3", "NVDA"],
        "market": ["US"] * 5,
        "industry": ["semi"] * 5,
        "close_usd": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    pit = pd.DataFrame([
        {"ticker": "NVDA", "available_date": "2024-05-01", "period_end": "2024-03-31", "ttm_eps": 20.0},
        {"ticker": "P1", "available_date": "2024-05-01", "period_end": "2024-03-31", "ttm_eps": 4.0},
        {"ticker": "P2", "available_date": "2024-05-01", "period_end": "2024-03-31", "ttm_eps": 5.0},
        {"ticker": "P3", "available_date": "2024-05-01", "period_end": "2024-03-31", "ttm_eps": 6.0},
    ])
    # 1. Without raw_close_usd or splits_df, refuses to fabricate split-corrupted US ep_yield
    m_safe = fpf.merge_pit_into_panel(panel_adj_only, pit)
    assert "pit_ep_z" not in m_safe.columns or m_safe["pit_ep_z"].isna().all()

    # 2. With splits_df (10-for-1 on 2024-06-10), NVDA true E/P = 2% < peers (4%,5%,6%) -> negative z-score!
    splits = pd.DataFrame([{"ticker": "NVDA", "split_date": "2024-06-10", "split_ratio": 10.0}])
    m_split = fpf.merge_pit_into_panel(panel_adj_only, pit, splits_df=splits)
    nvda_z = m_split.loc[(m_split["ticker"] == "NVDA") & (m_split["date"] == pd.Timestamp("2024-05-10")), "pit_ep_z"].iloc[0]
    assert nvda_z < 0.0, f"NVDA 2% E/P should rank BELOW 4-6% peers after split adjustment, got z={nvda_z}"

    # 3. With nominal raw_close_usd ($1000 pre-split, $100 post-split), NVDA E/P is ALSO exact (negative z-score!)
    panel_with_raw = panel_adj_only.copy()
    panel_with_raw["raw_close_usd"] = [1000.0, 100.0, 100.0, 100.0, 100.0]
    m_raw = fpf.merge_pit_into_panel(panel_with_raw, pit, splits_df=splits)
    nvda_z_raw = m_raw.loc[(m_raw["ticker"] == "NVDA") & (m_raw["date"] == pd.Timestamp("2024-05-10")), "pit_ep_z"].iloc[0]
    assert nvda_z_raw < 0.0, f"NVDA 2% E/P via raw_close_usd should be negative z-score, got {nvda_z_raw}"
