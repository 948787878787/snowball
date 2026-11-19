"""Offline unit tests for Snowball Global Multi-Market v5 Ranker, Backtester, and Extensible Universe DB."""

import numpy as np

import global_ranker as gr
import global_universe as gu
from global_universe import MARKET_CONFIG, get_market_groups, get_market_tickers


def test_all_global_market_presets_have_250_plus_and_20_per_industry():
    regional = ("US", "EU", "JP", "HK", "TW")
    total_regional = 0
    for mkt in regional:
        assert mkt in MARKET_CONFIG
        groups = get_market_groups(mkt)
        tickers = get_market_tickers(mkt)
        assert len(groups) >= 13, f"{mkt} expected >= 13 industries, got {len(groups)}"
        assert len(tickers) >= 250, f"{mkt} expected >= 250 stocks, got {len(tickers)}"
        assert len(tickers) == len(set(tickers)), f"{mkt} contains duplicate tickers"
        for ind, members in groups.items():
            assert len(members) >= 20, f"{mkt}.{ind} expected >= 20 stocks, got {len(members)}"
        total_regional += len(tickers)

    global_tickers = get_market_tickers("GLOBAL")
    assert len(global_tickers) >= 1250
    assert len(global_tickers) == total_regional


def test_global_v5_ranking_and_staggered_backtest_offline():
    for mkt in ("US", "EU", "JP"):
        groups = get_market_groups(mkt)
        close_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
        pf = gr.compute_global_price_factors(close_d, bench_d, groups)
        ranked = gr.score_latest_global(pf, fund_df, groups)

        assert len(ranked) == len(close_d.columns)
        assert len(ranked) >= 250
        assert ranked["v5_score"].is_monotonic_decreasing
        assert float(ranked["v5_score"].abs().max()) <= 3.0 + 1e-9

        bt = gr.backtest_global_staggered(pf, round_trip_cost=MARKET_CONFIG[mkt]["round_trip_cost"])
        assert bt["Weeks"] > 50
        assert np.isfinite(bt["Port_CAGR"])
        assert np.isfinite(bt["True_MDD"])
        assert bt["True_MDD"] <= 0.0


def test_industry_filter_and_custom_universe_extensibility(tmp_path):
    # Industry filtering
    semi = get_market_groups("US", industry="semiconductors_hardware")
    assert list(semi.keys()) == ["semiconductors_hardware"]
    assert len(semi["semiconductors_hardware"]) >= 20

    # Extensibility: add new market KR and extra tickers to US in an isolated custom directory
    try:
        path_kr, added_kr, total_kr = gu.add_tickers_to_universe(
            market="KR",
            industry="semiconductors_hardware",
            tickers=["005930.KS", "000660.KS", "005930.KS"],  # duplicate ignored
            target_dir=tmp_path,
            benchmark="^KS11",
            currency="KRW",
            name="South Korea (KRX)",
        )
        assert path_kr.exists()
        assert added_kr == 2
        assert total_kr == 2

        gu.reload_universes(custom_dir=tmp_path)
        assert "KR" in gu.MARKET_CONFIG
        assert set(gu.get_market_tickers("KR")) == {"005930.KS", "000660.KS"}
        assert "005930.KS" in gu.get_market_tickers("GLOBAL")

        # Also test merging extra tickers into an existing market
        _, added_us, _ = gu.add_tickers_to_universe(
            market="US",
            industry="software_internet",
            tickers=["APP", "MDB"],
            target_dir=tmp_path,
        )
        assert added_us == 2
        gu.reload_universes(custom_dir=tmp_path)
        us_soft = gu.get_market_groups("US", industry="software_internet")["software_internet"]
        assert "APP" in us_soft and "MDB" in us_soft
    finally:
        # Restore clean built-in registry state
        gu.reload_universes()


def test_custom_tickers_scoring():
    groups = {"custom": ("AAPL", "MSFT", "7203.T", "ASML.AS", "2330.TW", "0700.HK")}
    close_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
    pf = gr.compute_global_price_factors(close_d, bench_d, groups)
    ranked = gr.score_latest_global(pf, fund_df, groups)
    assert len(ranked) == 6
    assert set(ranked["ticker"]) == set(groups["custom"])
