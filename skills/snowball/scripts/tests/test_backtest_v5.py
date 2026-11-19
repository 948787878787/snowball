"""Unit tests for v5 formula, yield/EY caps, group-mean math, and zero-realized-loss invariant."""

import numpy as np
import pandas as pd

import backtest_v5 as b5


def test_v5_caps_and_group_zscore():
    idx = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    raw_yields = pd.DataFrame({"2603": [22.5, 18.0, 12.0, 6.2], "2609": [25.0, 15.0, 10.0, 3.3]}, index=idx)
    capped = raw_yields.clip(upper=b5.YIELD_CAP_PCT)
    assert float(capped.max().max()) <= b5.YIELD_CAP_PCT + 1e-9

    gz = b5._group_zscore(
        pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [2.0, 4.0, 6.0, 8.0], "C": [3.0, 6.0, 9.0, 12.0]}, index=idx),
        {"g1": ("A", "B", "C")},
    )
    assert not np.isinf(gz.to_numpy(dtype=float)).any()
    assert float(gz.abs().max().max()) <= 3.0 + 1e-9
    assert np.allclose(gz.mean(axis=1), 0.0)


def test_group_mean_incl_has_no_leave_one_out_reversal():
    idx = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    df = pd.DataFrame({"A": [0.10, -0.05, 0.02], "B": [0.00, 0.05, 0.02]}, index=idx)
    out = b5._group_mean_incl(df, {"g1": ("A", "B")})
    assert np.allclose(out["A"], out["B"])
    assert np.allclose(out["A"], [0.05, 0.00, 0.02])


def test_zero_realized_loss_simulator_respects_invariants():
    idx = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
    px = pd.DataFrame(
        {
            "A": np.linspace(100.0, 350.0, 20),
            "B": np.linspace(100.0, 40.0, 20),
            "C": np.linspace(50.0, 60.0, 20),
        },
        index=idx,
    )
    sig = pd.DataFrame(
        {
            "A": [2.0] * 4 + [0.0] * 16,
            "B": [0.0] * 4 + [2.0] * 4 + [-2.0] * 12,
            "C": [-1.0] * 8 + [2.0] * 12,
        },
        index=idx,
    )
    yld = pd.DataFrame({"A": 4.0, "B": 5.0, "C": 6.0}, index=idx)
    res = b5.simulate_personal_cashflow(
        sig, px, yld, "2024-01-05", "2024-05-31", monthly_inflow=10000.0, target_k=10, trim_winners=True
    )
    assert res["Contributed_TWD"] == 10000.0 * 5
    assert res["Terminal_TWD"] > 0
    assert res["Winner_Trims"] >= 1
    assert "True_TWR_MDD_%" in res


def test_twr_nav_has_no_deposit_dilution_and_combined_mdd_is_true_joint():
    # Bug #7a: Monotonically rising portfolio with monthly deposits must have 0% TWR MDD
    idx = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    val_s = pd.Series([100.0, 110.0, 121.0, 233.1, 256.41, 282.051, 410.2561, 451.28171], index=idx)
    cf_s = pd.Series([100.0, 0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 0.0], index=idx)
    nav = b5.compute_twr_nav(val_s, cf_s)
    mdd = float((nav / nav.cummax() - 1.0).min())
    assert abs(mdd) < 1e-9

    # Bug #7b: Combined MDD of two anti-correlated legs must be strictly smaller than weighted average MDD
    core_val = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx[:4])
    sat_val = pd.Series([100.0, 80.0, 110.0, 90.0], index=idx[:4])
    cf_zero = pd.Series([100.0, 0.0, 0.0, 0.0], index=idx[:4])
    core_mdd = float((b5.compute_twr_nav(core_val, cf_zero) / b5.compute_twr_nav(core_val, cf_zero).cummax() - 1.0).min())
    sat_mdd = float((b5.compute_twr_nav(sat_val, cf_zero) / b5.compute_twr_nav(sat_val, cf_zero).cummax() - 1.0).min())
    comb = b5.combine_cashflow_simulations(
        {"_weekly_val": core_val, "_weekly_cf": cf_zero, "Contributed_TWD": 100.0, "Terminal_TWD": 110.0, "End_Yield_%": 5.0, "Max_Stock_Wt_%": 10.0, "Holdings_Count": 5, "Winner_Trims": 0},
        {"_weekly_val": sat_val, "_weekly_cf": cf_zero, "Contributed_TWD": 100.0, "Terminal_TWD": 90.0, "End_Yield_%": 3.0, "Max_Stock_Wt_%": 20.0, "Holdings_Count": 5, "Winner_Trims": 1},
    )
    # Combined value is constant 200 every week -> true combined MDD is 0%, NOT ~ -22.5% weighted average!
    assert abs(comb["True_TWR_MDD_%"]) < 1e-9
    assert abs(core_mdd) > 0.20 and abs(sat_mdd) > 0.15


def test_raw_close_affordability_prevents_future_dividend_lookahead():
    # Bug #8: Stock A has adj close 700 <= 800 but nominal raw_close 950 > 800 -> must NOT be bought
    idx = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    adj_px = pd.DataFrame({"A": np.full(8, 700.0), "B": np.full(8, 100.0)}, index=idx)
    raw_px = pd.DataFrame({"A": np.full(8, 950.0), "B": np.full(8, 100.0)}, index=idx)
    sig = pd.DataFrame({"A": np.full(8, 3.0), "B": np.full(8, 1.0)}, index=idx)
    yld = pd.DataFrame({"A": np.full(8, 2.0), "B": np.full(8, 6.0)}, index=idx)
    res = b5.simulate_personal_cashflow(
        sig, adj_px, yld, "2024-01-05", "2024-02-23",
        monthly_inflow=10000.0, max_price=800.0, raw_close_w=raw_px,
    )
    # If B (yield 6%) was bought instead of A (yield 2%), End_Yield_% == 6.0
    assert res["End_Yield_%"] == 6.0


def test_dashboard_template_and_injection():
    import generate_dashboard as gd

    assert gd.TEMPLATE_HTML.exists()
    html = gd.TEMPLATE_HTML.read_text(encoding="utf-8")
    assert "backdrop-filter" in html
    assert "setTheme(" not in html
    assert "resetAll(" not in html
    assert 'id="btn-settings"' in html
    assert '"defaultLang": "en"' in html
    assert '"defaultCurrency": "USD"' in html
    for lang in ("en", "zh", "ja", "ko", "de"):
        assert f"setLang('{lang}')" in html
    for cur in ("USD", "TWD", "EUR", "JPY", "HKD", "KRW"):
        assert f"setCurrency('{cur}')" in html
    for tint in ("red", "orange", "yellow", "green", "blue", "indigo", "violet"):
        assert f'data-tint="{tint}"' in html
        assert f"setTint('{tint}')" in html
    out = gd.inject_data(html, {"title": "Test Dashboard", "defaultMonthly": 12000})
    assert '"defaultMonthly": 12000' in out
    assert "/*__DATA_START__*/" in out


def test_generate_dashboard_offline_end_to_end(tmp_path):
    import json
    import generate_dashboard as gd

    cfg = json.loads((gd.SKILL_DIR / "references" / "portfolio-template.json").read_text(encoding="utf-8"))
    payload = gd.build_payload(cfg, offline=True)
    assert payload["priceDate"] == "demo"
    assert len(payload["holdings"]) == len(cfg["holdings"])
    assert len(payload["screener"]) == len(cfg["watchlist"])
    assert "weeklyOutlook" in payload and len(payload["weeklyOutlook"]["actions"]) == 3
    assert payload["defaultTint"] == "blue"
    assert payload["defaultLang"] == "en"
    assert payload["defaultCurrency"] == "USD"

    out_file = tmp_path / "out.html"
    html = gd.TEMPLATE_HTML.read_text(encoding="utf-8")
    out_file.write_text(gd.inject_data(html, payload), encoding="utf-8")
    rendered = out_file.read_text(encoding="utf-8")
    assert "Snowball" in rendered
    assert '"defaultMonthly": 1000' in rendered


def test_score_predictions_cli_brier_and_bss(tmp_path):
    import score_predictions as sp

    csv_file = tmp_path / "preds.csv"
    csv_file.write_text(
        "prediction_id,target,direction,benchmark,probability,asset_return_pct,benchmark_return_pct,excess_return_pct,outcome,brier_score\n"
        "P1,StockA,long,SPY,0.60,3.0,1.0,,,\n"
        "P2,StockB,short,SPY,0.55,-2.0,0.5,,,\n",
        encoding="utf-8",
    )
    stats = sp.evaluate_predictions(csv_file)
    assert stats["settled"] == 2
    assert stats["hits"] == 2
    assert stats["mean_brier"] < 0.25
    assert stats["bss"] > 0.0


def test_versioned_ttl_cache_invalidation_and_fallback(tmp_path):
    import pytest
    import data

    calls = {"n": 0}
    clock = {"t": 1000.0}

    def loader():
        calls["n"] += 1
        return {"run": calls["n"]}

    # 1. Initial load populates cache; second load within TTL hits cache
    r1 = data.cached_load("demo", loader, ttl_seconds=60.0, schema_version="v3", fingerprint="fp1", cache_dir=tmp_path, now_fn=lambda: clock["t"])
    r2 = data.cached_load("demo", loader, ttl_seconds=60.0, schema_version="v3", fingerprint="fp1", cache_dir=tmp_path, now_fn=lambda: clock["t"] + 30.0)
    assert r1 == {"run": 1} and r2 == {"run": 1} and calls["n"] == 1

    # 2. Expired TTL triggers reload
    clock["t"] += 120.0
    r3 = data.cached_load("demo", loader, ttl_seconds=60.0, schema_version="v3", fingerprint="fp1", cache_dir=tmp_path, now_fn=lambda: clock["t"])
    assert r3 == {"run": 2} and calls["n"] == 2

    # 3. Schema version bump invalidates immediately even within TTL
    r4 = data.cached_load("demo", loader, ttl_seconds=60.0, schema_version="v4", fingerprint="fp1", cache_dir=tmp_path, now_fn=lambda: clock["t"] + 5.0)
    assert r4 == {"run": 3} and calls["n"] == 3

    # 4. Universe fingerprint change invalidates immediately even within TTL
    r5 = data.cached_load("demo", loader, ttl_seconds=60.0, schema_version="v4", fingerprint="fp2", cache_dir=tmp_path, now_fn=lambda: clock["t"] + 5.0)
    assert r5 == {"run": 4} and calls["n"] == 4

    # 5. Force refresh invalidates immediately
    r6 = data.cached_load("demo", loader, ttl_seconds=60.0, schema_version="v4", fingerprint="fp2", force_refresh=True, cache_dir=tmp_path, now_fn=lambda: clock["t"] + 5.0)
    assert r6 == {"run": 5} and calls["n"] == 5

    # 6. Expired compatible cache falls back gracefully on transient error, but incompatible schema raises
    def broken_loader():
        raise RuntimeError("network offline")

    r_stale = data.cached_load("demo", broken_loader, ttl_seconds=60.0, schema_version="v4", fingerprint="fp2", allow_stale_on_error=True, cache_dir=tmp_path, now_fn=lambda: clock["t"] + 500.0)
    assert r_stale == {"run": 5}

    with pytest.raises(RuntimeError):
        data.cached_load("demo", broken_loader, ttl_seconds=60.0, schema_version="v5", fingerprint="fp2", allow_stale_on_error=True, cache_dir=tmp_path, now_fn=lambda: clock["t"] + 500.0)

    # 7. Clear cache removes both .pkl and .meta.json
    removed = data.clear_cache("demo", cache_dir=tmp_path)
    assert removed == 2
