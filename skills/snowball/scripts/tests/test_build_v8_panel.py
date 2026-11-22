"""Integration test for v8 panel assembly on a small synthetic market (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import build_v8_panel as b8

CFG = {
    "dimensions": {
        "prior": {"pred_baseline_v6_prior": 1},
        "chip": {"chip_foreign_flow_20d": 1, "chip_margin_chg_4w": -1},
        "quality": {"q_gm_yoy": 1},
        "macro": {"macro_transmission": 1},
        "risk": {"risk_idio_vol_60d": -1},
    },
    "dimension_weights": {"prior": 0.2, "chip": 0.2, "quality": 0.2, "macro": 0.2, "risk": 0.2},
    "zscore": {"by": "date", "winsor": 3.0, "min_names": 3},
}


def _synthetic():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2021-01-01", "2024-06-28")
    tickers = [f"{1100 + i}.TW" for i in range(8)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (len(days), 8)), axis=0), index=days, columns=tickers)
    vol = pd.DataFrame(1e6, index=days, columns=tickers)
    weeks = pd.date_range("2022-01-07", "2024-06-28", freq="W-FRI")
    panel = pd.DataFrame([(d, t) for d in weeks for t in tickers], columns=["date", "ticker"])
    for c in ("f_mom_qual", "f_ind_mom", "f_low_risk", "f_rev_1w", "macro_stress"):
        panel[c] = rng.normal(size=len(panel))
    ids = [t.split(".")[0] for t in tickers]
    inst = pd.DataFrame([(i, d, rng.normal() * 1e4, rng.normal() * 1e3) for d in days for i in ids],
                        columns=["stock_id", "date", "foreign_net", "trust_net"])
    fr = pd.date_range("2021-01-01", "2024-06-28", freq="W-FRI")
    margin = pd.DataFrame([(i, d, rng.integers(100, 200)) for d in fr for i in ids], columns=["stock_id", "date", "margin_lots"])
    hold = pd.DataFrame([(i, d, rng.uniform(10, 40)) for d in fr for i in ids], columns=["stock_id", "date", "foreign_hold_pct"])
    stmts = pd.DataFrame([(i, y, s, 100.0, rng.uniform(10, 40), 5.0, 200.0, 100.0)
                          for i in ids for y in (2020, 2021, 2022, 2023, 2024) for s in (1, 2, 3, 4)],
                         columns=["stock_id", "year", "season", "revenue", "gross_profit", "net_income", "total_assets", "equity"])
    market = pd.DataFrame({
        "bench": close.mean(axis=1), "usd": 30 + np.cumsum(rng.normal(0, 0.1, len(days))),
        "us10y": 3 + np.cumsum(rng.normal(0, 0.03, len(days))), "oil": 70 + np.cumsum(rng.normal(0, 0.5, len(days))),
        "spx": 4000 + np.cumsum(rng.normal(0, 20, len(days))),
    }, index=days)
    import tw_mops_financials as tmf
    return panel, {"close": close, "raw_close": close, "volume": vol}, {"institutional": inst, "margin": margin, "foreign_hold": hold}, tmf.quality_factors(stmts), market


def test_assemble_adds_all_v8_columns_without_changing_rows():
    panel, daily, chip, quality, market = _synthetic()
    out = b8.assemble(panel, daily, chip, quality, market, CFG)
    assert len(out) == len(panel)
    for col in ("chip_foreign_flow_20d", "chip_trust_flow_20d", "chip_margin_chg_4w", "chip_foreign_hold_chg_12w",
                "risk_idio_vol_60d", "risk_max_21d", "risk_amihud_20d", "macro_transmission",
                "q_gm_yoy", "q_roa_yoy", "q_asset_growth", "pred_baseline_v6_prior", "v8_zero"):
        assert col in out.columns, col
    late = out[out["date"] >= "2024-01-01"]
    assert late[["chip_foreign_flow_20d", "risk_idio_vol_60d", "macro_transmission", "q_gm_yoy", "v8_zero"]].notna().mean().min() > 0.9


def test_coverage_table_has_one_row_per_year():
    panel, daily, chip, quality, market = _synthetic()
    cov = b8.coverage_by_year(b8.assemble(panel, daily, chip, quality, market, CFG), CFG)
    assert list(cov.index) == [2022, 2023, 2024]
