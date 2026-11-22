"""TDD tests for Snowball v7 T2: point-in-time TW universe, delisting settlement and survivorship-free mask."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stage5_validation as s5
import tw_pit_universe as tpu


def _daily_frame(start: str, end: str, tickers, value: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    return pd.DataFrame(value, index=idx, columns=list(tickers))


# --- universe selection --------------------------------------------------------------------------


def test_select_common_stocks_keeps_listed_four_digit_and_maps_suffix_and_group():
    info = pd.DataFrame({
        "stock_id": ["2330", "2330", "6488", "1234", "00878", "9999"],
        "industry_category": ["半導體業", "電子工業", "半導體業", "食品工業", "ETF", "其他"],
        "stock_name": ["a", "a", "b", "c", "d", "e"],
        "type": ["twse", "twse", "tpex", "emerging", "twse", "twse"],
        "date": ["2026-11-21"] * 6,
    })
    out = tpu.select_common_stocks(info)
    assert list(out["ticker"]) == ["2330.TW", "6488.TWO", "9999.TW"]
    grp = dict(zip(out["ticker"], out["group"]))
    assert grp["2330.TW"] == "semiconductors_hardware"
    assert grp["6488.TWO"] == "semiconductors_hardware"
    assert grp["9999.TW"] == "other"


def test_filter_delistings_keeps_common_stocks_after_start():
    raw = pd.DataFrame({
        "date": ["2018-05-01", "2021-03-02", "2022-01-05", "2023-04-21"],
        "stock_id": ["1111", "2448", "00732", "4144"],
        "stock_name": ["x", "y", "etf", "z"],
    })
    out = tpu.filter_delistings(raw, start="2019-09-01")
    assert list(out["stock_id"]) == ["2448", "4144"]
    assert out["delist_date"].dtype.kind == "M"


# --- membership mask -----------------------------------------------------------------------------


def _mask_inputs():
    tickers = ["A", "B", "C"]
    close = _daily_frame("2024-01-01", "2024-03-29", tickers)
    vol = close.copy()
    vol["A"], vol["B"], vol["C"] = 3_000.0, 2_000.0, 1_000.0
    # In February C becomes the most liquid name.
    feb = (vol.index >= "2024-02-01") & (vol.index <= "2024-02-29")
    vol.loc[feb, "C"] = 9_000.0
    weeks = pd.date_range("2024-01-05", "2024-03-29", freq="W-FRI")
    return close, vol, weeks


def test_mask_selection_at_month_end_only_affects_later_weeks():
    close, vol, weeks = _mask_inputs()
    mask = tpu.build_monthly_membership_mask(close, vol, weeks, top_n=2, lookback_days=10, min_obs=5)
    # Jan month-end (2024-01-31): top-2 = A, B. Effective for weeks in (Jan 31, Feb 29].
    feb_weeks = [w for w in weeks if pd.Timestamp("2024-01-31") < w <= pd.Timestamp("2024-02-29")]
    for w in feb_weeks:
        assert mask.loc[w, "A"] and mask.loc[w, "B"] and not mask.loc[w, "C"]
    # Feb month-end selection (C liquid) only applies strictly after 2024-02-29.
    mar_weeks = [w for w in weeks if w > pd.Timestamp("2024-02-29")]
    for w in mar_weeks:
        assert mask.loc[w, "C"]
    # Weeks before the first month-end have no prior selection -> nobody is a member.
    jan_weeks = [w for w in weeks if w <= pd.Timestamp("2024-01-31")]
    assert not mask.loc[jan_weeks].to_numpy().any()


def test_mask_has_no_lookahead_to_post_month_end_volume():
    close, vol, weeks = _mask_inputs()
    base = tpu.build_monthly_membership_mask(close, vol, weeks, top_n=2, lookback_days=10, min_obs=5)
    shocked = vol.copy()
    shocked.loc[shocked.index > "2024-01-31", "C"] = 1e12  # future liquidity spike
    alt = tpu.build_monthly_membership_mask(close, shocked, weeks, top_n=2, lookback_days=10, min_obs=5)
    feb_weeks = [w for w in weeks if pd.Timestamp("2024-01-31") < w <= pd.Timestamp("2024-02-29")]
    pd.testing.assert_frame_equal(base.loc[feb_weeks], alt.loc[feb_weeks])


def test_delisted_stock_never_appears_after_last_trade():
    close, vol, weeks = _mask_inputs()
    close.loc[close.index > "2024-02-14", "A"] = np.nan  # A stops trading mid-Feb
    vol.loc[vol.index > "2024-02-14", "A"] = np.nan
    mask = tpu.build_monthly_membership_mask(close, vol, weeks, top_n=2, lookback_days=10, min_obs=5)
    assert mask.loc[pd.Timestamp("2024-02-09"), "A"]
    for w in weeks[weeks > pd.Timestamp("2024-02-16")]:
        assert not mask.loc[w, "A"]


# --- delisting settlement ------------------------------------------------------------------------


def test_settlement_fills_next_open_with_final_close_and_flags_final_week():
    weeks = pd.date_range("2024-01-05", "2024-02-23", freq="W-FRI")
    next_open = pd.DataFrame({"X": [10.0, 11.0, 12.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                              "Y": [5.0] * 8}, index=weeks)
    close_w = pd.DataFrame({"X": [10.0, 11.0, 12.0, 9.0, np.nan, np.nan, np.nan, np.nan],
                            "Y": [5.0] * 8}, index=weeks)
    filled, final = tpu.apply_delisting_settlement(next_open, close_w, {"X": False})
    # Week of last trade (index 3) settles at the final close 9.0; afterwards it stays flat.
    assert filled["X"].iloc[3] == pytest.approx(9.0)
    assert filled["X"].iloc[-1] == pytest.approx(9.0)
    assert bool(final.loc[weeks[3], "X"]) and int(final["X"].sum()) == 1
    # Non-delisted tickers untouched.
    pd.testing.assert_series_equal(filled["Y"], next_open["Y"])
    assert not final["Y"].any()


def test_settlement_applies_haircut_when_last_price_is_stale():
    weeks = pd.date_range("2024-01-05", "2024-01-26", freq="W-FRI")
    next_open = pd.DataFrame({"X": [10.0, np.nan, np.nan, np.nan]}, index=weeks)
    close_w = pd.DataFrame({"X": [10.0, 8.0, np.nan, np.nan]}, index=weeks)
    filled, _ = tpu.apply_delisting_settlement(next_open, close_w, {"X": True})
    assert filled["X"].iloc[1] == pytest.approx(8.0 * (1.0 + tpu.STALE_DELIST_HAIRCUT))


# --- portfolio simulator honours realised delisting settlement ----------------------------------


def test_simulator_does_not_double_penalise_settled_delisting():
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "ticker": "Y.TW", "market": "TW", "industry": "x", "score": 0.1, "fwd_ret_1w_open": 0.0})
        if i <= 1:
            rows.append({
                "date": d, "ticker": "X.TW", "market": "TW", "industry": "x", "score": 0.9,
                "fwd_ret_1w_open": -0.10 if i == 0 else 0.0,
                "delist_final": i == 1,
            })
    df = pd.DataFrame(rows)
    df["delist_final"] = df["delist_final"].fillna(False).astype(bool)
    weekly = s5.compute_portfolio_weekly_returns(
        df, "score", top_k=1, rebalance_every=4, market="TW",
        max_industry_weight=None, buffer_mult=1.0, neutralize_scores=False,
    )
    # Only the realised -10% hits; no extra -30% after the settled final week.
    assert weekly.min() == pytest.approx(-0.10, abs=0.02)
    assert (weekly > -0.25).all()



# --- panel assembly helpers ----------------------------------------------------------------------


def test_rerank_targets_uses_only_members_in_cross_section():
    import build_tw_pit_panel as bpp

    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-05")] * 3,
        "ticker": ["A", "B", "C"],
        "market": ["TW"] * 3,
        "fwd_vol_adj_4w": [0.1, 0.2, 0.3],
        "target_rank_4w": [0.33, 0.66, 1.0],
    })
    out = bpp.rerank_targets(df.iloc[:2], horizons=(4,))
    assert list(out["target_rank_4w"]) == [0.5, 1.0]
    assert list(out["target_rank_mkt_4w"]) == [0.5, 1.0]


def test_merge_groups_keeps_existing_and_adds_new_tickers():
    import build_tw_pit_panel as bpp

    base = {"banks": ("2881.TW",), "semiconductors_hardware": ("2330.TW",)}
    extra = pd.DataFrame({"ticker": ["2330.TW", "6488.TWO", "9999.TW"], "group": ["software_internet", "semiconductors_hardware", "other"]})
    out = bpp.merge_groups(base, extra)
    assert "2330.TW" in out["semiconductors_hardware"] and "2330.TW" not in out.get("software_internet", ())
    assert "6488.TWO" in out["semiconductors_hardware"]
    assert out["other"] == ("9999.TW",)


def test_combine_price_sources_prefers_yahoo_and_unions_dates():
    import build_tw_pit_panel as bpp

    y = pd.DataFrame({"A.TW": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    f = pd.DataFrame({"A.TW": [9.0], "D.TW": [5.0]}, index=pd.to_datetime(["2024-01-04"]))
    out = bpp.combine_price_sources(y, f)
    assert list(out.columns) == ["A.TW", "D.TW"]
    assert np.isnan(out.loc["2024-01-04", "A.TW"])  # Yahoo-covered names never take FinMind values
    assert out.loc["2024-01-04", "D.TW"] == 5.0


# --- T2 report ----------------------------------------------------------------------------------


def test_yearly_prior_ic_reports_per_year_and_horizon():
    import t2_universe_report as rep

    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2022-01-07", "2023-12-29", freq="W-FRI"):
        mom = rng.normal(size=30)
        for i in range(30):
            rows.append({
                "date": d, "ticker": f"T{i}", "market": "TW", "macro_stress": 0.0,
                "f_mom_qual": mom[i], "f_ind_mom": 0.0, "f_low_risk": 0.0, "f_rev_1w": 0.0,
                "target_rank_4w": mom[i], "target_rank_12w": -mom[i],
            })
    out = rep.yearly_prior_ic(pd.DataFrame(rows), horizons=(4, 12))
    assert list(out.index) == [2022, 2023]
    assert out["ic_4w"].tolist() == pytest.approx([1.0, 1.0])
    assert out["ic_12w"].tolist() == pytest.approx([-1.0, -1.0])
    assert (out["n_weeks"] == 52).all()


def test_settlement_uses_last_trade_date_when_prices_are_forward_filled():
    weeks = pd.date_range("2024-01-05", "2024-02-02", freq="W-FRI")
    # USD conversion forward-fills, so close/next_open never go NaN after delisting.
    next_open = pd.DataFrame({"X": [10.0, 11.0, 12.0, 12.0, 12.0]}, index=weeks)
    close_w = pd.DataFrame({"X": [10.0, 11.0, 9.0, 9.0, 9.0]}, index=weeks)
    filled, final = tpu.apply_delisting_settlement(
        next_open, close_w, {"X": False}, last_trade={"X": pd.Timestamp("2024-01-17")},
    )
    assert bool(final.loc[weeks[2], "X"]) and int(final["X"].sum()) == 1
    assert filled["X"].iloc[2:].tolist() == pytest.approx([9.0, 9.0, 9.0])
    assert filled["X"].iloc[:2].tolist() == pytest.approx([10.0, 11.0])


def test_etf_codes_starting_with_zero_are_not_common_stocks():
    info = pd.DataFrame({"stock_id": ["0050", "1101"], "industry_category": ["ETF", "水泥工業"],
                         "stock_name": ["x", "y"], "type": ["twse", "twse"], "date": ["2026-11-21"] * 2})
    assert list(tpu.select_common_stocks(info)["ticker"]) == ["1101.TW"]
    raw = pd.DataFrame({"date": ["2022-05-27", "2022-06-01"], "stock_id": ["0054", "2448"], "stock_name": ["a", "b"]})
    assert list(tpu.filter_delistings(raw, start="2019-01-01")["stock_id"]) == ["2448"]


def test_next_open_wide_blanks_prices_after_delisting_final_week():
    import build_tw_pit_panel as bpp

    d = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    frame = pd.DataFrame({
        "date": list(d) * 2,
        "ticker": ["X"] * 4 + ["Y"] * 4,
        "next_open_usd": [1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0],
        "delist_final": [False, True, False, False] + [False] * 4,
    })
    wide = bpp.next_open_wide(frame)
    assert wide["X"].tolist()[:2] == [1.0, 2.0]
    assert wide["X"].iloc[2:].isna().all()
    assert wide["Y"].notna().all()
