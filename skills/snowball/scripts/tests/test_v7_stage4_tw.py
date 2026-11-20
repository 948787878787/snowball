"""Stage 4 TDD Tests for Low-Risk Taiwan Feature Expansion (Snowball v7).

Tests Stage 4.1, 4.2, and 4.3 from `Snowball_v7_計劃與邏輯_給Gemini.md`:
- 4.1 Taiwan monthly revenue PIT rule (available_date = next month 11th), 3m cumulative YoY
      (clipped [-0.80, +2.00]), YoY acceleration (clipped [-1.00, +1.00]), 12m revenue high,
      and strict backward merge with zero look-ahead.
- 4.2 52-week high proximity (George & Hwang 2004): close / max(close, 52w), strictly causal.
- 4.3 4w + 12w horizon rank blending: 0.5 * rank(4w) + 0.5 * rank(12w).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tw_monthly_rev as tmr


def test_tw_monthly_revenue_pit_available_date_is_next_month_11th():
    raw = pd.DataFrame([
        {"ticker": "2330.TW", "revenue_year": 2024, "revenue_month": 5, "revenue": 200.0},
        {"ticker": "2330.TW", "revenue_year": 2024, "revenue_month": 12, "revenue": 250.0},
    ])
    feat = tmr.compute_tw_monthly_rev_features(raw)
    avail = feat["available_date"].dt.strftime("%Y-%m-%d").tolist()
    assert avail == ["2024-06-11", "2025-01-11"], f"Expected next-month 11th PIT rule, got {avail}"


def test_tw_monthly_revenue_features_3m_yoy_accel_and_12m_high():
    # Build 28 months of synthetic monthly revenue for 2330.TW (2022-01 .. 2024-04)
    # Year 1 (2022): 100/mo; Year 2 (2023): 120/mo (+20% YoY); Early 2024: 180/mo (+50% YoY -> accel = +30%)
    rows = []
    for y, val in [(2022, 100.0), (2023, 120.0)]:
        for m in range(1, 13):
            rows.append({"ticker": "2330.TW", "revenue_year": y, "revenue_month": m, "revenue": val})
    for m in range(1, 5):
        rows.append({"ticker": "2330.TW", "revenue_year": 2024, "revenue_month": m, "revenue": 180.0})

    # Also test clipping: extreme +500% jump must clip at +2.00; extreme accel clips at +1.00
    for y, val in [(2022, 10.0), (2023, 10.0), (2024, 100.0)]:
        for m in range(1, 13 if y < 2024 else 5):
            rows.append({"ticker": "9999.TW", "revenue_year": y, "revenue_month": m, "revenue": val})

    feat = tmr.compute_tw_monthly_rev_features(pd.DataFrame(rows))
    tsmc_last = feat[(feat["ticker"] == "2330.TW") & (feat["revenue_year"] == 2024) & (feat["revenue_month"] == 4)].iloc[0]
    assert abs(tsmc_last["tw_rev_3m_yoy"] - 0.50) < 1e-6, f"Expected 3m YoY=0.50, got {tsmc_last['tw_rev_3m_yoy']}"
    assert abs(tsmc_last["tw_rev_yoy_accel"] - 0.20) < 1e-6 or tsmc_last["tw_rev_yoy_accel"] > 0.15
    assert tsmc_last["tw_rev_12m_high"] == 1.0

    ext_last = feat[(feat["ticker"] == "9999.TW") & (feat["revenue_year"] == 2024) & (feat["revenue_month"] == 4)].iloc[0]
    assert ext_last["tw_rev_3m_yoy"] == 2.0, "3m YoY must clip at +2.00"
    assert ext_last["tw_rev_yoy_accel"] <= 1.0, "YoY accel must clip at +1.00"


def test_tw_monthly_revenue_merge_enforces_zero_lookahead():
    # May 2024 revenue -> available 2024-06-11.
    # On 2024-06-07 (before 11th) must be NaN; on 2024-06-14 (after 11th) must be populated!
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-07"] * 4 + ["2024-06-14"] * 4).astype("datetime64[ns]"),
        "ticker": ["A.TW", "B.TW", "C.TW", "D.TW"] * 2,
        "market": ["TW"] * 8,
        "industry": ["semi"] * 8,
    })
    rev_feat = pd.DataFrame([
        {"ticker": "A.TW", "available_date": pd.Timestamp("2024-06-11"), "tw_rev_3m_yoy": 0.50, "tw_rev_yoy_accel": 0.20, "tw_rev_12m_high": 1.0},
        {"ticker": "B.TW", "available_date": pd.Timestamp("2024-06-11"), "tw_rev_3m_yoy": 0.10, "tw_rev_yoy_accel": 0.00, "tw_rev_12m_high": 0.0},
        {"ticker": "C.TW", "available_date": pd.Timestamp("2024-06-11"), "tw_rev_3m_yoy": -0.10, "tw_rev_yoy_accel": -0.10, "tw_rev_12m_high": 0.0},
        {"ticker": "D.TW", "available_date": pd.Timestamp("2024-06-11"), "tw_rev_3m_yoy": 0.30, "tw_rev_yoy_accel": 0.10, "tw_rev_12m_high": 1.0},
    ])
    merged = tmr.merge_tw_monthly_rev_into_panel(panel, rev_feat)
    pre = merged.loc[(merged["ticker"] == "A.TW") & (merged["date"] == pd.Timestamp("2024-06-07")), "pit_tw_rev_3m_yoy_z"].iloc[0]
    post = merged.loc[(merged["ticker"] == "A.TW") & (merged["date"] == pd.Timestamp("2024-06-14")), "pit_tw_rev_3m_yoy_z"].iloc[0]
    assert pd.isna(pre), "Look-ahead violation: May revenue leaked before June 11!"
    assert np.isfinite(post) and post > 0.0


def test_52w_high_proximity_george_hwang_causal():
    dates = pd.date_range("2023-01-06", periods=60, freq="W-FRI")
    rows = []
    for i, dt in enumerate(dates):
        # Stock A steadily climbs to new 52w high (ratio=1.0); Stock B drops 50% from peak (ratio=0.5)
        rows.append({"date": dt, "ticker": "A.TW", "market": "TW", "industry": "semi", "close_usd": 10.0 + i})
        rows.append({"date": dt, "ticker": "B.TW", "market": "TW", "industry": "semi", "close_usd": 100.0 if i < 20 else 50.0})
        rows.append({"date": dt, "ticker": "C.TW", "market": "TW", "industry": "semi", "close_usd": 80.0})
        rows.append({"date": dt, "ticker": "D.TW", "market": "TW", "industry": "semi", "close_usd": 60.0 + 0.5 * i})
    out = tmr.add_52w_high_proximity(pd.DataFrame(rows))
    last_dt = dates[-1]
    a_z = out.loc[(out["ticker"] == "A.TW") & (out["date"] == last_dt), "f_52w_high"].iloc[0]
    b_z = out.loc[(out["ticker"] == "B.TW") & (out["date"] == last_dt), "f_52w_high"].iloc[0]
    assert a_z > b_z, f"Stock at 52w high must score higher than stock 50% off peak: A={a_z}, B={b_z}"

    # Verify zero look-ahead: mutating future price on last week must not change earlier weeks
    df_mut = pd.DataFrame(rows)
    df_mut.loc[(df_mut["ticker"] == "A.TW") & (df_mut["date"] == last_dt), "close_usd"] = 9999.0
    out_mut = tmr.add_52w_high_proximity(df_mut)
    mid_dt = dates[40]
    assert abs(
        out.loc[(out["ticker"] == "A.TW") & (out["date"] == mid_dt), "f_52w_high"].iloc[0]
        - out_mut.loc[(out_mut["ticker"] == "A.TW") & (out_mut["date"] == mid_dt), "f_52w_high"].iloc[0]
    ) < 1e-6


def test_blend_4w_12w_predictions():
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    oos_4w = pd.DataFrame([
        {"date": d, "ticker": t, "market": "TW", "industry": "semi", "fwd_ret_1w_open": 0.01, "pred_m": sc}
        for d in dates
        for t, sc in [("A.TW", 10.0), ("B.TW", 5.0), ("C.TW", 1.0), ("D.TW", 0.0)]
    ])
    oos_12w = pd.DataFrame([
        {"date": d, "ticker": t, "market": "TW", "industry": "semi", "pred_m": sc}
        for d in dates
        for t, sc in [("A.TW", 1.0), ("B.TW", 10.0), ("C.TW", 5.0), ("D.TW", 0.0)]
    ])
    blended = tmr.blend_horizon_predictions(oos_4w, oos_12w, score_col="pred_m", w_4w=0.5)
    assert "pred_m_blend_4w_12w" in blended.columns
    # A has rank 1.0 in 4w and 0.5 in 12w -> avg 0.75; B has rank 0.75 in 4w and 1.0 in 12w -> avg 0.875 (B > A > D)
    s0 = blended[blended["date"] == dates[0]].set_index("ticker")["pred_m_blend_4w_12w"]
    assert s0["B.TW"] > s0["A.TW"] > s0["D.TW"]
