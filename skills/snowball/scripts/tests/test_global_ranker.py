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
        assert ranked["v6_score"].is_monotonic_decreasing
        assert float(ranked["v6_score"].abs().max()) <= 3.0 + 1e-9
        assert {"mom_qual", "fip", "ind_mom", "norm_val", "quality", "low_risk", "v5_score", "v6_score"}.issubset(ranked.columns)

        bt_v6 = gr.backtest_global_staggered(pf, round_trip_cost=MARKET_CONFIG[mkt]["round_trip_cost"], model="v6")
        assert bt_v6["Weeks"] > 50
        assert bt_v6["Model"] == "v6"
        assert bt_v6["Execution"] == "Next-Day Open"
        assert np.isfinite(bt_v6["Port_CAGR"])
        assert np.isfinite(bt_v6["True_MDD"])
        assert bt_v6["True_MDD"] <= 0.0

        bt_v5 = gr.backtest_global_staggered(pf, round_trip_cost=MARKET_CONFIG[mkt]["round_trip_cost"], model="v5")
        assert bt_v5["Model"] == "v5"
        assert np.isfinite(bt_v5["Port_CAGR"])


def test_v6_fip_quality_and_low_risk_math():
    # Construct 4 stocks over 160 trading days (~32 weeks):
    # SMOOTH: rises steadily +1% every week (smooth Frog-in-the-Pan winner, low IVOL, low Beta)
    # JUMPY : same ~22% total return via 1 lottery jump +25% and tiny weekly declines (bad FIP, high IVOL)
    # VALUE_TRAP: low P/E but terrible ROE/margins/negative FCF and high leverage
    # QUALITY_VAL: cheap EV/EBITDA + high ROE/margins/positive FCF + low debt
    dates = pd.bdate_range("2023-01-02", periods=320)
    rng = np.random.default_rng(7)
    mkt_r = rng.normal(0.0003, 0.010, len(dates))
    bench = pd.Series(100.0 * np.cumprod(1.0 + mkt_r), index=dates)

    # Smooth stock: steady positive drift, low beta 0.5, tiny noise
    r_smooth = 0.5 * mkt_r + 0.0012 + rng.normal(0.0, 0.002, len(dates))
    # Jumpy stock: high beta 1.8, mostly tiny negative days plus 2 big lottery spikes
    r_jumpy = 1.8 * mkt_r - 0.0008 + rng.normal(0.0, 0.015, len(dates))
    r_jumpy[210] += 0.35
    r_jumpy[240] += 0.25

    close_d = pd.DataFrame({
        "SMOOTH": 100.0 * np.cumprod(1.0 + r_smooth),
        "JUMPY": 100.0 * np.cumprod(1.0 + r_jumpy),
        "QUAL": 100.0 * np.cumprod(1.0 + 0.8 * mkt_r + 0.0005),
        "TRAP": 100.0 * np.cumprod(1.0 + 1.2 * mkt_r + 0.0005),
    }, index=dates)
    groups = {"g1": ("SMOOTH", "JUMPY", "QUAL", "TRAP")}
    pf = gr.compute_global_price_factors(close_d, bench, groups, open_d=close_d)

    fund = pd.DataFrame([
        {"ticker": "SMOOTH", "name": "Smooth", "dividend_yield": 2.0, "PER": 15.0, "PBR": 2.0, "ev_ebitda": 10.0, "rev_growth": 0.12, "earnings_growth": 0.15, "roe": 0.20, "op_margin": 0.20, "gross_margin": 0.50, "fcf_yield": 0.05, "debt_to_equity": 30.0},
        {"ticker": "JUMPY",  "name": "Jumpy",  "dividend_yield": 2.0, "PER": 15.0, "PBR": 2.0, "ev_ebitda": 10.0, "rev_growth": 0.12, "earnings_growth": 0.15, "roe": 0.20, "op_margin": 0.20, "gross_margin": 0.50, "fcf_yield": 0.05, "debt_to_equity": 30.0},
        {"ticker": "QUAL",   "name": "Qual",   "dividend_yield": 3.0, "PER": 12.0, "PBR": 2.0, "ev_ebitda": 7.0,  "rev_growth": 0.10, "earnings_growth": 0.12, "roe": 0.32, "op_margin": 0.28, "gross_margin": 0.65, "fcf_yield": 0.09, "debt_to_equity": 15.0},
        {"ticker": "TRAP",   "name": "Trap",   "dividend_yield": 3.0, "PER": 12.0, "PBR": 2.0, "ev_ebitda": 25.0, "rev_growth": -0.10,"earnings_growth": -0.15,"roe": 0.01, "op_margin": 0.02, "gross_margin": 0.12, "fcf_yield": -0.05,"debt_to_equity": 280.0},
    ]).set_index("ticker")

    ranked = gr.score_latest_global(pf, fund, groups, model="v6").set_index("ticker")
    assert ranked.loc["SMOOTH", "fip"] > ranked.loc["JUMPY", "fip"]
    assert ranked.loc["SMOOTH", "low_risk"] > ranked.loc["JUMPY", "low_risk"]
    assert ranked.loc["QUAL", "quality"] > ranked.loc["TRAP", "quality"]
    assert ranked.loc["QUAL", "norm_val"] > ranked.loc["TRAP", "norm_val"]
    assert ranked.loc["QUAL", "v6_score"] > ranked.loc["TRAP", "v6_score"]


def test_macro_regime_conditioning_and_micro_lerner_markup():
    import macro_micro as mm

    # 1. Dynamic weights stay bounded, sum to 1.0, and tilt economically across regimes
    w_base = mm.regime_conditioned_weights(0.0, 0.0)
    w_stress = mm.regime_conditioned_weights(1.2, 0.0)
    w_infl = mm.regime_conditioned_weights(0.0, 1.2)
    w_exp = mm.regime_conditioned_weights(-1.0, -0.5)
    for w in (w_base, w_stress, w_infl, w_exp):
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert all(v > 0 for v in w.values())
    # Credit stress tilts into Quality & Low-Risk and away from Momentum
    assert w_stress["quality"] > w_base["quality"]
    assert w_stress["low_risk"] > w_base["low_risk"]
    assert w_stress["mom_qual"] < w_base["mom_qual"]
    # Rate/inflation shock tilts into Short-Duration Value & Pricing-Power Quality and away from Growth
    assert w_infl["norm_val"] > w_base["norm_val"]
    assert w_infl["quality"] > w_base["quality"]
    assert w_infl["fund"] < w_base["fund"]
    # Expansion tilts toward Momentum
    assert w_exp["mom_qual"] > w_base["mom_qual"]

    # 2. Microeconomic Lerner excess markup rewards above-peer gross margin
    gm = pd.Series({"A": 0.70, "B": 0.40, "C": 0.35, "D": 0.20})
    markup = mm.compute_micro_lerner_markup(gm, {"g1": ("A", "B", "C", "D")})
    assert markup["A"] > 0 and markup["D"] < 0
    assert markup["A"] > markup["B"] > markup["C"] > markup["D"]

    # 3. Macro panel strict shift(1) anti-look-ahead: last-bar shock does NOT leak into same-bar regime
    dates = pd.bdate_range("2022-01-03", periods=400)
    macro_d = mm.make_offline_macro_data(dates)
    weeks = macro_d.resample("W-FRI").last().index
    p1 = mm.compute_macro_regime_panel(macro_d, weeks)
    macro_shocked = macro_d.copy()
    macro_shocked.iloc[-1] = macro_shocked.iloc[-1] * 3.0  # shock on final trading day only
    p2 = mm.compute_macro_regime_panel(macro_shocked, weeks)
    assert abs(float(p1["macro_stress"].iloc[-1]) - float(p2["macro_stress"].iloc[-1])) < 1e-12

    # 4. Macro-conditioned backtest runs cleanly end-to-end
    groups = {"g1": ("A", "B", "C", "D", "E", "F", "G", "H")}
    close_d, open_d, bench_d, _ = gr.make_offline_demo_data(groups)
    pf = gr.compute_global_price_factors(close_d, bench_d, groups, open_d=open_d)
    mp = mm.compute_macro_regime_panel(mm.make_offline_macro_data(close_d.index), pf["close_w"].index)
    bt = gr.backtest_global_staggered(pf, round_trip_cost=0.001, model="v6", macro_panel=mp)
    assert np.isfinite(bt["Port_CAGR"]) and bt["Model"] == "v6"


def test_sequoia_rule_of_40_rewards_capital_efficient_growth():
    dates = pd.bdate_range("2023-01-02", periods=220)
    mkt_r = np.full(len(dates), 0.0004)
    bench = pd.Series(100.0 * np.cumprod(1.0 + mkt_r), index=dates)
    close_d = pd.DataFrame(
        {"EFFICIENT": 100.0 * np.cumprod(1.0 + mkt_r), "BURNER": 100.0 * np.cumprod(1.0 + mkt_r)},
        index=dates,
    )
    groups = {"software_internet": ("EFFICIENT", "BURNER")}
    pf = gr.compute_global_price_factors(close_d, bench, groups, open_d=close_d)
    fund = pd.DataFrame([
        {"ticker": "EFFICIENT", "name": "Eff", "dividend_yield": 0.0, "PER": 25.0, "PBR": 5.0, "ev_ebitda": 18.0, "rev_growth": 0.20, "earnings_growth": 0.20, "roe": 0.20, "op_margin": 0.20, "gross_margin": 0.70, "fcf_yield": 0.04, "fcf_margin": 0.25, "debt_to_equity": 30.0, "current_ratio": 2.0},
        {"ticker": "BURNER",    "name": "Burn","dividend_yield": 0.0, "PER": 25.0, "PBR": 5.0, "ev_ebitda": 18.0, "rev_growth": 0.20, "earnings_growth": 0.20, "roe": 0.20, "op_margin": 0.20, "gross_margin": 0.70, "fcf_yield": 0.04, "fcf_margin": -0.15,"debt_to_equity": 30.0, "current_ratio": 2.0},
    ]).set_index("ticker")

    ranked = gr.score_latest_global(pf, fund, groups, model="v6").set_index("ticker")
    assert abs(ranked.loc["EFFICIENT", "rule40_%"] - 45.0) < 1e-6
    assert abs(ranked.loc["BURNER", "rule40_%"] - 5.0) < 1e-6
    assert ranked.loc["EFFICIENT", "v6_score"] > ranked.loc["BURNER", "v6_score"]


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
