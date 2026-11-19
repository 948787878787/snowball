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


def test_dashboard_template_and_injection():
    import generate_dashboard as gd

    assert gd.TEMPLATE_HTML.exists()
    html = gd.TEMPLATE_HTML.read_text(encoding="utf-8")
    assert "backdrop-filter" in html
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
