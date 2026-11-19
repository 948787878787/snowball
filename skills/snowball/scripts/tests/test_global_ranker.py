"""Offline unit tests for Snowball Global Multi-Market v5 Ranker, Backtester, and Code-Review Fixes."""

import json

import numpy as np
import pandas as pd

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

    assert len(get_market_tickers("US")) >= 1000
    assert len(get_market_tickers("TW")) >= 1000
    global_tickers = get_market_tickers("GLOBAL")
    assert len(global_tickers) >= 2780
    assert len(global_tickers) == total_regional


def test_global_v5_ranking_and_staggered_backtest_offline():
    for mkt in ("US", "EU", "JP"):
        groups = get_market_groups(mkt)
        close_d, open_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
        pf = gr.compute_global_price_factors(close_d, bench_d, groups, open_d=open_d)
        ranked = gr.score_latest_global(pf, fund_df, groups)

        assert len(ranked) == len(close_d.columns)
        assert len(ranked) >= 250
        assert ranked["v5_score"].is_monotonic_decreasing
        assert float(ranked["v5_score"].abs().max()) <= 3.0 + 1e-9

        bt = gr.backtest_global_staggered(pf, round_trip_cost=MARKET_CONFIG[mkt]["round_trip_cost"])
        assert bt["Weeks"] > 50
        assert bt["Execution"] == "Next-Day Open"
        assert np.isfinite(bt["Port_CAGR"])
        assert np.isfinite(bt["True_MDD"])
        assert bt["True_MDD"] <= 0.0


def test_dividend_yield_and_trailing_pe_parsers():
    # Bug #2: Yahoo 2025 dividendYield=0.42 means 0.42%, NEVER 42% (capped to 8%)
    aapl_full = {
        "dividendYield": 0.42,
        "trailingAnnualDividendYield": 0.0042,
        "trailingAnnualDividendRate": 1.00,
        "regularMarketPrice": 238.10,
    }
    assert abs(gr.parse_yahoo_dividend_yield_pct(aapl_full) - 0.42) < 0.02

    aapl_trailing_only = {"dividendYield": 0.42, "trailingAnnualDividendYield": 0.0042}
    assert abs(gr.parse_yahoo_dividend_yield_pct(aapl_trailing_only) - 0.42) < 1e-6

    aapl_dy_only = {"dividendYield": 0.42}
    assert abs(gr.parse_yahoo_dividend_yield_pct(aapl_dy_only) - 0.42) < 1e-6

    # Bug #6: Loss-making firm must NOT fall back to forwardPE
    loss_firm = {"trailingPE": None, "trailingEps": -2.5, "forwardPE": 15.0}
    assert np.isnan(gr.parse_yahoo_trailing_pe(loss_firm))

    neg_eps_firm = {"trailingPE": 12.0, "trailingEps": -0.5, "forwardPE": 10.0}
    assert np.isnan(gr.parse_yahoo_trailing_pe(neg_eps_firm))

    profitable_firm = {"trailingPE": 18.5, "trailingEps": 6.4, "forwardPE": 16.0}
    assert gr.parse_yahoo_trailing_pe(profitable_firm) == 18.5


def test_standalone_universe_file_is_reachable(tmp_path):
    # Bug #4: --universe-file without --market must rank ONLY the file, not 1,340 GLOBAL stocks
    ufile = tmp_path / "mini.json"
    ufile.write_text(
        json.dumps({
            "market": "MINI",
            "name": "Mini Custom",
            "benchmark": "SPY",
            "currency": "USD",
            "round_trip_cost": 0.001,
            "industries": {"tech": ["AAPL", "MSFT", "NVDA"]},
        }),
        encoding="utf-8",
    )
    groups, tickers, bench, cost, label, _, _ = gr.resolve_universe_selection(universe_file=ufile)
    assert set(tickers) == {"AAPL", "MSFT", "NVDA"}
    assert bench == "SPY"
    assert cost == 0.001
    assert label == "MINI"


def test_global_fx_normalization_removes_currency_drift():
    # Bug #5: JPY stock doubling locally purely due to JPY weakening 100 -> 200 has 0% USD return
    dates = pd.bdate_range("2024-01-02", periods=10)
    jpy_per_usd = pd.Series(np.linspace(100.0, 200.0, 10), index=dates)
    close_d = pd.DataFrame(
        {"7203.T": np.linspace(1000.0, 2000.0, 10), "AAPL": np.full(10, 150.0)},
        index=dates,
    )
    close_usd, _ = gr.convert_prices_to_usd(close_d, fx_series_map={"JPY": (jpy_per_usd, False)})
    assert np.allclose(close_usd["7203.T"], 10.0)
    assert np.allclose(close_usd["AAPL"], 150.0)


def test_next_day_open_execution_differs_from_same_bar_close():
    # Bug #3: Entering at a +5% Monday open gap must produce strictly lower CAGR than Friday-close fill
    groups = {"g1": ("A", "B", "C", "D", "E", "F", "G", "H")}
    close_d, _, bench_d, _ = gr.make_offline_demo_data(groups)
    open_flat = close_d.copy()
    open_gap_up = close_d * 1.05
    pf_flat = gr.compute_global_price_factors(close_d, bench_d, groups, open_d=open_flat)
    pf_gap = gr.compute_global_price_factors(close_d, bench_d, groups, open_d=open_gap_up)
    bt_flat = gr.backtest_global_staggered(pf_flat, round_trip_cost=0.001)
    bt_gap = gr.backtest_global_staggered(pf_gap, round_trip_cost=0.001)
    assert bt_gap["Port_CAGR"] < bt_flat["Port_CAGR"]


def test_industry_filter_and_custom_universe_extensibility(tmp_path):
    semi = get_market_groups("US", industry="semiconductors_hardware")
    assert list(semi.keys()) == ["semiconductors_hardware"]
    assert len(semi["semiconductors_hardware"]) >= 20

    try:
        path_kr, added_kr, total_kr = gu.add_tickers_to_universe(
            market="KR",
            industry="semiconductors_hardware",
            tickers=["005930.KS", "000660.KS", "005930.KS"],
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
        gu.reload_universes()


def test_custom_tickers_scoring():
    groups = {"custom": ("AAPL", "MSFT", "7203.T", "ASML.AS", "2330.TW", "0700.HK")}
    close_d, open_d, bench_d, fund_df = gr.make_offline_demo_data(groups)
    pf = gr.compute_global_price_factors(close_d, bench_d, groups, open_d=open_d)
    ranked = gr.score_latest_global(pf, fund_df, groups)
    assert len(ranked) == 6
    assert set(ranked["ticker"]) == set(groups["custom"])
