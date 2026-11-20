"""Stage 1 TDD Tests for Snowball v7 (Credible Baseline).

Covers every Stage 1 requirement from `Snowball_v7_計劃與邏輯_給Gemini.md`:
- 1.1 P0-6: XBRL multi-tag union (tag switching), 10-K Q4 TTM derivation, dei shares + nominal raw_close
      split-immune E/P, 200-day staleness tolerance
- 1.2 Survivorship bias `membership_mask` integration & conservative -30% delisting return
- 1.3 Realistic per-market cost & tax model (TW 0.585% / odd-lot 1.185%, US 0.15% + 30% div withholding,
      limit-up/down lock handling, true per-rebalance turnover)
- 1.4 Purge gap = (horizon + 1) weeks
- Principle #1 (52-week holdout isolation) & Principle #2 (trial_log.csv + Deflated Sharpe Ratio)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_training_panel as btp
import evaluate as ev
import fetch_pit_fundamentals as fpf
import train_walkforward as twf


def test_xbrl_tag_switch_continues_to_latest_filings():
    """Company switches from legacy SalesRevenueNet (more rows) to ASC-606 tag (newer rows)."""
    legacy_rows = [
        {
            "form": "10-K",
            "start": f"{y}-01-01",
            "end": f"{y}-12-31",
            "filed": f"{y + 1}-02-15",
            "val": 1000.0 + y,
        }
        for y in range(2010, 2019)
    ]
    new_rows = [
        {"form": "10-K", "start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-15", "val": 5000.0},
        {"form": "10-K", "start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-15", "val": 6000.0},
    ]
    payload = {
        "facts": {
            "us-gaap": {
                "SalesRevenueNet": {"units": {"USD": legacy_rows}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": new_rows}},
            }
        }
    }
    out = fpf.parse_sec_company_facts("AAPL", payload, min_year=2018)
    assert not out.empty
    assert out["available_date"].max() >= "2025-02-16", (
        f"Tag switch froze series at legacy tag! max available_date={out['available_date'].max()}"
    )
    latest = out.iloc[-1]
    assert abs(latest["rev_yoy"] - 0.20) < 1e-6


def test_ttm_derives_q4_from_10k_when_q4_not_separately_tagged():
    """Most US filers report Q1, Q2, Q3 in 10-Q and full-year in 10-K without a standalone Q4 tag."""
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "start": "2023-01-01", "end": "2023-03-31", "filed": "2023-05-01", "val": 100.0},
                            {"form": "10-Q", "start": "2023-04-01", "end": "2023-06-30", "filed": "2023-08-01", "val": 100.0},
                            {"form": "10-Q", "start": "2023-07-01", "end": "2023-09-30", "filed": "2023-11-01", "val": 100.0},
                            {"form": "10-K", "start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-15", "val": 420.0},
                            {"form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 150.0},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "start": "2023-01-01", "end": "2023-03-31", "filed": "2023-05-01", "val": 10.0},
                            {"form": "10-Q", "start": "2023-04-01", "end": "2023-06-30", "filed": "2023-08-01", "val": 10.0},
                            {"form": "10-Q", "start": "2023-07-01", "end": "2023-09-30", "filed": "2023-11-01", "val": 10.0},
                            {"form": "10-K", "start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-15", "val": 45.0},
                            {"form": "10-Q", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 25.0},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "start": "2024-03-31", "end": "2024-03-31", "filed": "2024-05-01", "val": 200.0},
                        ]
                    }
                },
            }
        }
    }
    out = fpf.parse_sec_company_facts("TEST", payload, min_year=2023)
    # In Q1 2024 (filed 2024-05-01): TTM NI = FY2023 (45) + Q1_2024 (25) - Q1_2023 (10) = 60 -> ROE = 60 / 200 = 0.30
    q1_24 = out[out["available_date"] == "2024-05-02"].iloc[0]
    assert abs(q1_24["roe"] - 0.30) < 1e-6, f"Expected derived Q4 TTM ROE=0.30, got {q1_24['roe']}"
    assert "ttm_net_income" in out.columns and abs(q1_24["ttm_net_income"] - 60.0) < 1e-6


def test_market_cap_ep_uses_dei_shares_and_nominal_raw_close():
    """TTM NetIncome / (dei shares * nominal raw_close) is 100% split-immune across a 10:1 split."""
    payload = {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"form": "10-Q", "end": "2024-03-31", "filed": "2024-05-01", "val": 1e8},
                            {"form": "10-Q", "end": "2024-06-30", "filed": "2024-08-01", "val": 1e9},
                        ]
                    }
                }
            },
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "start": "2023-04-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 1e10},
                            {"form": "10-Q", "start": "2024-04-01", "end": "2024-06-30", "filed": "2024-08-01", "val": 2.5e9},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "start": "2023-04-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 2e9},
                            {"form": "10-Q", "start": "2023-04-01", "end": "2023-06-30", "filed": "2023-08-01", "val": 5e8},
                            {"form": "10-Q", "start": "2024-04-01", "end": "2024-06-30", "filed": "2024-08-01", "val": 5e8},
                        ]
                    }
                },
            },
        }
    }
    pit_nvda = fpf.parse_sec_company_facts("NVDA", payload, min_year=2023)
    assert "shares_out" in pit_nvda.columns
    assert "ttm_net_income" in pit_nvda.columns

    # Pre-split: 100M shares @ nominal $1000 = $100B mkt cap -> E/P = 2B / 100B = 2.0%
    # Post-split: 1B shares @ nominal $100 = $100B mkt cap -> E/P = 2B / 100B = 2.0% (ratio = 1.00!)
    ep_pre = pit_nvda.iloc[0]["ttm_net_income"] / (pit_nvda.iloc[0]["shares_out"] * 1000.0)
    ep_post = pit_nvda.iloc[-1]["ttm_net_income"] / (pit_nvda.iloc[-1]["shares_out"] * 100.0)
    ratio = ep_pre / ep_post
    assert 0.80 <= ratio <= 1.25, f"Pre/post split E/P ratio distorted: {ratio}"


def test_default_staleness_tolerance_is_200_days():
    assert fpf.DEFAULT_MAX_STALENESS_DAYS == 200


def test_purge_gap_requires_horizon_plus_one_weeks_and_freezes_holdout():
    panel = btp.build_combined_panel(
        markets=("US", "TW"),
        horizons=(4,),
        offline_demo=True,
        demo_stocks_per_market=20,
        demo_days=900,
    )
    horizon = 4
    cutoff = "2024-11-22"
    res = twf.run_purged_walkforward(
        panel=panel,
        horizon=horizon,
        min_train_weeks=40,
        refit_step_weeks=8,
        enable_lgbm=False,
        top_k=10,
        holdout_cutoff=cutoff,
    )
    # 1.4: Purge gap must be >= (horizon + 1) * 7 = 35 days
    for entry in res["purge_audit"]:
        assert entry["gap_days"] >= (horizon + 1) * 7, f"Purge gap < (h+1)*7: {entry}"
    # Principle #1: Holdout dates > cutoff MUST NOT appear anywhere in OOS predictions
    oos_max = pd.Timestamp(res["oos_predictions"]["date"].max())
    assert oos_max <= pd.Timestamp(cutoff), f"Holdout leaked! oos_max={oos_max} > {cutoff}"


def test_real_cost_model_separates_market_costs_us_div_tax_and_delisting():
    dates = pd.date_range("2024-01-05", periods=16, freq="W-FRI")
    rows = []
    for i, d in enumerate(dates):
        for j in range(10):
            # Force full turnover every 4 weeks by alternating score order
            sc = float(j if (i // 4) % 2 == 0 else (9 - j))
            rows.append({
                "date": d,
                "ticker": f"T{j}.TW",
                "market": "TW",
                "pred_m": sc,
                "fwd_ret_1w_open": 0.005,
            })
    df_tw = pd.DataFrame(rows)
    tw_board = twf._simulate_topk_twr(df_tw, "pred_m", top_k=4, rebalance_every=4, market="TW", tw_odd_lot=False)
    tw_odd = twf._simulate_topk_twr(df_tw, "pred_m", top_k=4, rebalance_every=4, market="TW", tw_odd_lot=True)
    assert "Avg_Rebalance_Turnover" in tw_board
    assert "Annual_Round_Trips" in tw_board
    assert tw_board["Avg_Rebalance_Turnover"] > 0.5
    assert tw_odd["TWR_CAGR"] < tw_board["TWR_CAGR"], "TW odd-lot slippage must reduce net CAGR vs board-lot"

    # US dividend withholding tax drag
    df_us = df_tw.copy()
    df_us["market"] = "US"
    df_us["ticker"] = [f"U{j}" for _ in dates for j in range(10)]
    us_no_tax = twf._simulate_topk_twr(df_us, "pred_m", top_k=4, rebalance_every=4, market="US", us_div_withhold_annual=0.0)
    us_with_tax = twf._simulate_topk_twr(df_us, "pred_m", top_k=4, rebalance_every=4, market="US", us_div_withhold_annual=0.0045)
    assert us_with_tax["TWR_CAGR"] < us_no_tax["TWR_CAGR"]

    # Delisting conservative -30% return when a held ticker disappears mid-holding
    df_delist = df_tw.copy()
    mask_delist = (df_delist["ticker"] == "T9.TW") & (df_delist["date"] >= dates[2])
    df_delist.loc[mask_delist, "fwd_ret_1w_open"] = np.nan
    res_delist = twf._simulate_topk_twr(df_delist, "pred_m", top_k=4, rebalance_every=4, market="TW", delist_return=-0.30)
    assert res_delist["TWR_CAGR"] < tw_board["TWR_CAGR"] - 0.02


def test_membership_mask_filters_non_members_in_walkforward():
    panel = btp.build_combined_panel(
        markets=("US",),
        horizons=(4,),
        offline_demo=True,
        demo_stocks_per_market=20,
        demo_days=860,
    )
    dates = sorted(panel["date"].unique())
    tickers = sorted(panel["ticker"].unique())
    mask = pd.DataFrame(True, index=pd.DatetimeIndex(dates), columns=tickers)
    excluded_ticker = tickers[0]
    mask[excluded_ticker] = False

    res = twf.run_purged_walkforward(
        panel=panel,
        horizon=4,
        min_train_weeks=40,
        refit_step_weeks=8,
        enable_lgbm=False,
        top_k=5,
        membership_mask=mask,
    )
    assert excluded_ticker not in set(res["oos_predictions"]["ticker"].unique())


def test_deflated_sharpe_ratio_and_trial_log(tmp_path: Path):
    dsr_1 = ev.deflated_sharpe_ratio(sharpe=1.0, n_trials=1, n_obs=200)
    dsr_20 = ev.deflated_sharpe_ratio(sharpe=1.0, n_trials=20, n_obs=200)
    assert 0.0 <= dsr_20 < dsr_1 <= 1.0, f"More trials must deflate DSR: dsr_1={dsr_1}, dsr_20={dsr_20}"

    log_path = tmp_path / "trial_log.csv"
    twf.append_trial_log(log_path, stage="stage1_baseline", market="TW", model="stage1_icir", change="initial", ic=0.05, sharpe=0.75, mdd=-0.25)
    twf.append_trial_log(log_path, stage="stage1_baseline", market="US", model="stage1_icir", change="initial", ic=-0.01, sharpe=0.20, mdd=-0.30)
    df_log = pd.read_csv(log_path)
    assert len(df_log) == 2
    assert df_log["trial_id"].tolist() == [1, 2]


def test_yahoo_split_adjusted_close_with_post_split_cover_shares_date():
    # WMT-style 3-for-1 split on 2024-02-26:
    # Yahoo auto_adjust=False Close is split-adjusted to panel_end ($60 both pre and post split).
    # Pre-split filing (period_end 2023-10-31, shares_date 2023-12-01) reports 2.7B pre-split shares -> scaled x3 = 8.1B.
    # Post-split 10-K (period_end 2024-01-31, filed 2024-03-15, shares_date 2024-03-14) ALREADY reports 8.1B post-split shares -> scaled x1 = 8.1B.
    # Both must produce identical Market Cap ($486B) and E/P (16.2B / 486B = 3.33%), NOT 3x jump!
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-12", "2024-04-12"]).astype("datetime64[ns]"),
        "ticker": ["WMT", "WMT"],
        "market": ["US", "US"],
        "industry": ["retail", "retail"],
        "raw_close_usd": [60.0, 60.0],
        "close_usd": [59.0, 60.0],
    })
    pit = pd.DataFrame([
        {
            "ticker": "WMT",
            "available_date": pd.Timestamp("2023-12-05"),
            "period_end": pd.Timestamp("2023-10-31"),
            "shares_date": pd.Timestamp("2023-12-01"),
            "ttm_net_income": 16.2e9,
            "shares_out": 2.7e9,
        },
        {
            "ticker": "WMT",
            "available_date": pd.Timestamp("2024-03-15"),
            "period_end": pd.Timestamp("2024-01-31"),
            "shares_date": pd.Timestamp("2024-03-14"),
            "ttm_net_income": 16.2e9,
            "shares_out": 8.1e9,
        },
    ])
    pit["available_date"] = pd.to_datetime(pit["available_date"]).astype("datetime64[ns]")
    pit["period_end"] = pd.to_datetime(pit["period_end"]).astype("datetime64[ns]")
    pit["shares_date"] = pd.to_datetime(pit["shares_date"]).astype("datetime64[ns]")
    splits = pd.DataFrame([{"ticker": "WMT", "split_date": "2024-02-26", "split_ratio": 3.0}])
    dbg = pd.merge_asof(
        panel.sort_values("date"),
        pit.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        by="ticker",
        direction="backward",
    )
    ep = fpf._compute_split_immune_us_ep(dbg, True, splits, False)
    assert abs(float(ep.iloc[0]) - float(ep.iloc[1])) < 1e-6, f"Expected identical E/P across WMT split, got {ep.tolist()}"
    assert abs(float(ep.iloc[0]) - (16.2e9 / (8.1e9 * 60.0))) < 1e-6

    # GOOGL-style ASC 260 retroactive restatement: period_end & shares_date (2022-06-30) are BEFORE
    # the 20:1 split (2022-07-18), but available_date (2022-07-28) is AFTER the split, so SEC already
    # reported post-split shares (13.0B). Must NOT multiply by 20 again!
    g_panel = pd.DataFrame({
        "date": pd.to_datetime(["2022-05-06", "2022-08-05"]).astype("datetime64[ns]"),
        "ticker": ["GOOGL", "GOOGL"],
        "market": ["US", "US"],
        "industry": ["tech", "tech"],
        "raw_close_usd": [110.0, 110.0],
    })
    g_pit = pd.DataFrame([
        {"ticker": "GOOGL", "available_date": "2022-04-28", "period_end": "2022-03-31", "shares_date": "2022-03-31", "ttm_net_income": 71.5e9, "shares_out": 6.5e8},
        {"ticker": "GOOGL", "available_date": "2022-07-28", "period_end": "2022-06-30", "shares_date": "2022-06-30", "ttm_net_income": 71.5e9, "shares_out": 13.0e9},
    ])
    for c in ("available_date", "period_end", "shares_date"):
        g_pit[c] = pd.to_datetime(g_pit[c]).astype("datetime64[ns]")
    g_splits = pd.DataFrame([{"ticker": "GOOGL", "split_date": "2022-07-18", "split_ratio": 20.0}])
    g_dbg = pd.merge_asof(g_panel.sort_values("date"), g_pit.sort_values("available_date"), left_on="date", right_on="available_date", by="ticker", direction="backward")
    g_ep = fpf._compute_split_immune_us_ep(g_dbg, True, g_splits, False)
    assert abs(float(g_ep.iloc[0]) - float(g_ep.iloc[1])) < 1e-6, f"GOOGL ASC 260 double-split bug: {g_ep.tolist()}"


def test_membership_mask_filters_in_build_combined_panel():
    base = btp.build_combined_panel(
        markets=("US",),
        horizons=(4,),
        offline_demo=True,
        demo_stocks_per_market=15,
        demo_days=400,
    )
    dates = sorted(base["date"].unique())
    tickers = sorted(base["ticker"].unique())
    mask = pd.DataFrame(True, index=pd.DatetimeIndex(dates), columns=tickers)
    dropped = tickers[0]
    mask[dropped] = False

    filtered = btp.build_combined_panel(
        markets=("US",),
        horizons=(4,),
        offline_demo=True,
        demo_stocks_per_market=15,
        demo_days=400,
        membership_mask=mask,
    )
    assert dropped not in set(filtered["ticker"].unique())
    assert len(filtered) < len(base)

