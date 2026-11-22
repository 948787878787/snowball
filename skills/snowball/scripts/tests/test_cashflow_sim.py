"""TDD tests for the user's real-rule cash-flow simulator (buy-only DCA, odd lots, zero realized loss)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cashflow_sim as cfs


def _panel(prices: dict, scores: dict, start: str = "2024-01-05", n: int = 10) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="W-FRI")
    rows = []
    for i, d in enumerate(dates):
        for t in prices:
            p = prices[t][i] if isinstance(prices[t], list) else prices[t]
            if p is None:
                continue
            s = scores[t][i] if isinstance(scores[t], list) else scores[t]
            rows.append({"date": d, "ticker": t, "px": p, "score": s})
    return pd.DataFrame(rows)


def _cfg(**kw) -> cfs.CashflowConfig:
    base = dict(monthly_cash=1000.0, n_buy=1, buy_pct=0.9, sell_pct=0.6, fee_rate=0.001425,
                min_fee=1.0, sell_tax=0.003, max_weight=1.0, allow_loss_sells=False)
    base.update(kw)
    return cfs.CashflowConfig(**base)


def test_contribution_added_once_per_calendar_month():
    df = _panel({"A": 100.0, "B": 50.0}, {"A": 0.9, "B": 0.1}, n=9)  # Jan 5 .. Mar 1 -> 3 months
    res = cfs.simulate(df, "score", "px", _cfg())
    assert res.contributions.sum() == pytest.approx(3000.0)
    assert (res.contributions > 0).sum() == 3


def test_buys_whole_odd_lot_shares_with_minimum_fee():
    df = _panel({"A": 300.0, "B": 50.0}, {"A": 0.9, "B": 0.1}, n=1)
    res = cfs.simulate(df, "score", "px", _cfg())
    # 1000 cash: shares = floor((1000 - fee) / 300) = 3; fee = max(1, 900 * 0.001425) = 1.2825
    assert res.final_positions["A"] == 3
    assert res.cash.iloc[-1] == pytest.approx(1000.0 - 900.0 - 1.2825)


def test_zero_loss_rule_blocks_selling_a_losing_position():
    # A bought at 100, falls to 80 while score drops into sell zone; must be kept.
    df = _panel({"A": [100.0, 80.0, 80.0], "B": 50.0}, {"A": [0.9, 0.0, 0.0], "B": [0.1, 0.9, 0.9]}, n=3)
    res = cfs.simulate(df, "score", "px", _cfg())
    assert res.final_positions.get("A", 0) > 0
    assert res.blocked_sells >= 1
    assert res.realized_pnl == pytest.approx(0.0)


def test_profitable_position_in_sell_zone_is_sold():
    df = _panel({"A": [100.0, 130.0, 130.0], "B": 50.0}, {"A": [0.9, 0.0, 0.0], "B": [0.1, 0.9, 0.9]}, n=3)
    res = cfs.simulate(df, "score", "px", _cfg())
    assert res.final_positions.get("A", 0) == 0
    assert res.realized_pnl > 0


def test_allow_loss_sells_flag_sells_losers():
    df = _panel({"A": [100.0, 80.0, 80.0], "B": 50.0}, {"A": [0.9, 0.0, 0.0], "B": [0.1, 0.9, 0.9]}, n=3)
    res = cfs.simulate(df, "score", "px", _cfg(allow_loss_sells=True))
    assert res.final_positions.get("A", 0) == 0
    assert res.realized_pnl < 0


def test_missing_price_for_held_name_is_written_down_and_removed():
    df = _panel({"A": [100.0, None, None], "B": 50.0}, {"A": [0.9, 0.9, 0.9], "B": [0.1, 0.1, 0.1]}, n=3)
    res = cfs.simulate(df, "score", "px", _cfg(), delist_haircut=-0.30)
    assert "A" not in res.final_positions
    assert res.delist_events == 1


def test_xirr_simple_one_year():
    flows = pd.Series([-100.0, 110.0], index=pd.to_datetime(["2024-01-01", "2024-12-31"]))
    assert cfs.xirr(flows) == pytest.approx(0.10, abs=2e-3)


def test_dca_benchmark_matches_buy_and_hold_math():
    dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
    px = pd.Series([10.0, 10.0, 10.0, 10.0, 20.0], index=dates)
    val, contrib = cfs.dca_benchmark(px, monthly_cash=100.0, fee_rate=0.0, min_fee=0.0)
    assert contrib.sum() == pytest.approx(200.0)  # Jan + Feb
    assert val.iloc[-1] == pytest.approx(10 * 20.0 + 5 * 20.0)  # Jan 10 sh @10, Feb 5 sh @20


def test_unsettled_tail_week_without_any_prices_is_not_a_delisting():
    df = _panel({"A": [100.0, 100.0, None], "B": [50.0, 50.0, None]}, {"A": 0.9, "B": 0.1}, n=3)
    tail = df["date"].max() + pd.Timedelta(days=7)
    df = pd.concat([df, pd.DataFrame({"date": [tail, tail], "ticker": ["A", "B"], "px": [float("nan")] * 2, "score": [0.9, 0.1]})])
    res = cfs.simulate(df, "score", "px", _cfg())
    assert res.delist_events == 0
    assert res.final_positions.get("A", 0) > 0


def test_held_name_leaving_universe_is_valued_from_fallback_not_written_down():
    df = _panel({"A": [100.0, None, None], "B": 50.0}, {"A": [0.9, 0.9, 0.9], "B": [0.1, 0.1, 0.1]}, n=3)
    dates = sorted(df["date"].unique())
    wide = pd.DataFrame({"A": [100.0, 120.0, 130.0], "B": [50.0, 50.0, 50.0]}, index=pd.DatetimeIndex(dates))
    res = cfs.simulate(df, "score", "px", _cfg(), price_fallback=wide)
    assert res.delist_events == 0
    assert res.final_positions.get("A", 0) > 0
    a_sh = res.final_positions["A"]
    assert res.values.iloc[-1] >= a_sh * 130.0


def test_name_missing_from_both_panel_and_fallback_is_liquidated_at_last_price():
    df = _panel({"A": [100.0, None, None], "B": 50.0}, {"A": [0.9, 0.9, 0.9], "B": [0.1, 0.1, 0.1]}, n=3)
    dates = sorted(df["date"].unique())
    wide = pd.DataFrame({"A": [100.0, float("nan"), float("nan")], "B": [50.0] * 3}, index=pd.DatetimeIndex(dates))
    res = cfs.simulate(df, "score", "px", _cfg(), price_fallback=wide, delist_haircut=0.0)
    assert res.delist_events == 1 and "A" not in res.final_positions


def test_weekly_twr_strips_contributions():
    idx = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    values = pd.Series([100.0, 210.0, 231.0], index=idx)
    contrib = pd.Series([100.0, 100.0, 0.0], index=idx)
    twr = cfs.weekly_twr(values, contrib)
    assert twr.tolist() == pytest.approx([0.10, 0.10])


# --- v8 lump-sum user rules: initial capital, per-buy amount, name / industry caps, odd-lot slippage ---


def _lump_cfg(**kw) -> cfs.CashflowConfig:
    return _cfg(monthly_cash=0.0, initial_cash=100_000.0, buy_amount=10_000.0, buy_pct=0.0, **kw)


def _ranked_panel(n_names: int = 6, n_weeks: int = 3, industries=None) -> pd.DataFrame:
    prices = {f"T{i}": 100.0 for i in range(n_names)}
    scores = {f"T{i}": float(n_names - i) for i in range(n_names)}
    df = _panel(prices, scores, n=n_weeks)
    if industries is not None:
        df["ind"] = df["ticker"].map(industries)
    return df


def test_lump_sum_is_contributed_once_and_buys_fixed_amount_per_week():
    df = _ranked_panel(n_weeks=3)
    res = cfs.simulate(df, "score", "px", _lump_cfg())
    assert res.contributions.sum() == pytest.approx(100_000.0)
    assert res.contributions.iloc[0] == pytest.approx(100_000.0)
    buys = [t for t in res.trade_log if t["side"] == "buy"]
    assert len(buys) == 3                                   # one name per week
    assert all(b["shares"] * b["px"] <= 10_000.0 for b in buys)
    assert res.cash.iloc[-1] > 69_000.0                     # the rest stays as cash


def test_max_names_cap_blocks_new_names_once_full():
    df = _ranked_panel(n_weeks=4)
    res = cfs.simulate(df, "score", "px", _lump_cfg(max_names=2, max_weight=0.10))
    assert len(res.final_positions) == 2


def test_industry_cap_skips_names_from_a_full_industry():
    inds = {"T0": "semi", "T1": "semi", "T2": "semi", "T3": "fin", "T4": "fin", "T5": "food"}
    df = _ranked_panel(n_weeks=3, industries=inds)
    res = cfs.simulate(df, "score", "px", _lump_cfg(industry_cap=0.15, max_weight=0.10), industry_col="ind")
    bought = [t["ticker"] for t in res.trade_log if t["side"] == "buy"]
    assert bought == ["T0", "T3", "T5"]                     # second semi would breach 15% of equity


def test_odd_lot_slippage_raises_buy_price_and_lowers_sell_price():
    df = _panel({"A": [100.0, 150.0], "B": 50.0}, {"A": [0.9, 0.1], "B": [0.5, 0.9]}, n=2)
    res = cfs.simulate(df, "score", "px", _cfg(monthly_cash=1000.0, buy_pct=0.0, sell_pct=0.6, slippage=0.003))
    buy, sell = [t for t in res.trade_log if t["side"] == "buy"][0], [t for t in res.trade_log if t["side"] == "sell"][0]
    assert buy["px"] == pytest.approx(100.3)
    assert sell["px"] == pytest.approx(150.0 * 0.997)


def test_lump_benchmark_buys_once_and_holds():
    px = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"))
    vals, contribs = cfs.lump_benchmark(px, initial_cash=10_000.0)
    assert contribs.sum() == pytest.approx(10_000.0)
    assert vals.iloc[-1] == pytest.approx(vals.iloc[0] - 99 * 100.0 + 99 * 121.0)
