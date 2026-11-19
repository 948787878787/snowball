"""Offline unit tests for Snowball Global Multi-Market v5 Ranker & Backtester."""

import numpy as np

import global_ranker as gr
from global_universe import MARKET_CONFIG, get_market_groups, get_market_tickers


def test_all_global_market_presets_exist_and_non_empty():
    for mkt in ("US", "EU", "JP", "HK", "TW", "GLOBAL"):
        assert mkt in MARKET_CONFIG
        tickers = get_market_tickers(mkt)
        assert len(tickers) >= 12


def test_global_v5_ranking_and_staggered_backtest_offline():
    for mkt in ("US", "GLOBAL"):
        groups = get_market_groups(mkt)
        close_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
        pf = gr.compute_global_price_factors(close_d, bench_d, groups)
        ranked = gr.score_latest_global(pf, fund_df, groups)

        assert len(ranked) == len(close_d.columns)
        assert ranked["v5_score"].is_monotonic_decreasing
        assert float(ranked["v5_score"].abs().max()) <= 3.0 + 1e-9

        bt = gr.backtest_global_staggered(pf, round_trip_cost=MARKET_CONFIG[mkt]["round_trip_cost"])
        assert bt["Weeks"] > 50
        assert np.isfinite(bt["Port_CAGR"])
        assert np.isfinite(bt["True_MDD"])
        assert bt["True_MDD"] <= 0.0


def test_custom_tickers_scoring():
    groups = {"custom": ("AAPL", "MSFT", "7203.T", "ASML.AS", "2330.TW", "0700.HK")}
    close_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
    pf = gr.compute_global_price_factors(close_d, bench_d, groups)
    ranked = gr.score_latest_global(pf, fund_df, groups)
    assert len(ranked) == 6
    assert set(ranked["ticker"]) == set(groups["custom"])
