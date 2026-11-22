"""TDD tests for the pre-registered v8 zero-parameter composite and frozen-config handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import v8_composite as vc

CFG = {
    "dimensions": {"a": {"x": 1, "y": -1}, "b": {"z": 1}},
    "dimension_weights": {"a": 0.5, "b": 0.5},
    "zscore": {"by": "date", "winsor": 3.0, "min_names": 3},
}


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "date": np.repeat(pd.to_datetime(["2025-01-03", "2025-01-10"]), n // 2),
        "ticker": [f"T{i}" for i in range(n)],
        "industry": np.tile(["semi", "fin"], n // 2),
        "x": rng.normal(size=n), "y": rng.normal(size=n), "z": rng.normal(size=n),
    })


def test_cs_zscore_is_standardised_and_winsorised_per_date():
    df = _frame()
    df.loc[0, "x"] = 1e6
    z = vc.cs_zscore(df, "x", winsor=3.0, min_names=3)
    assert z.max() <= 3.0 + 1e-12
    per_date = z.groupby(df["date"]).mean()
    assert np.allclose(per_date, 0.0, atol=0.2)


def test_single_extreme_outlier_does_not_compress_other_names():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"date": pd.Timestamp("2025-01-03"), "x": rng.normal(size=200)})
    df.loc[0, "x"] = -1e6
    z = vc.cs_zscore(df, "x", winsor=3.0, min_names=3, winsor_pct=0.01)
    assert z.drop(index=0).std() > 0.9
    assert z.loc[0] == z.min()


def test_cs_zscore_nan_when_too_few_names():
    df = _frame().head(2)
    assert vc.cs_zscore(df, "x", winsor=3.0, min_names=3).isna().all()


def test_dimension_score_applies_signs_and_weights():
    df = _frame()
    out = vc.add_v8_zero(df, CFG)
    za = (vc.cs_zscore(df, "x", 3.0, 3) - vc.cs_zscore(df, "y", 3.0, 3)) / 2
    assert np.allclose(out["dim_a"], za)
    expected = 0.5 * out["dim_a"] + 0.5 * out["dim_b"]
    assert np.allclose(out["v8_zero"], expected)


def test_missing_dimension_renormalises_weights_instead_of_zero_fill():
    df = _frame().assign(z=np.nan)
    out = vc.add_v8_zero(df, CFG)
    assert out["dim_b"].isna().all()
    assert np.allclose(out["v8_zero"], out["dim_a"])


def test_missing_input_column_raises_clear_error():
    with pytest.raises(KeyError, match="z"):
        vc.add_v8_zero(_frame().drop(columns="z"), CFG)


def test_industry_demean_zeroes_each_industry_mean():
    df = vc.add_v8_zero(_frame(), CFG)
    out = vc.industry_demean(df, "v8_zero")
    assert np.allclose(out.groupby([df["date"], df["industry"]]).mean(), 0.0)


def test_config_hash_stable_under_key_order_and_detects_edits(tmp_path: Path):
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    p1.write_text(json.dumps({"a": 1, "b": [1, 2]}))
    p2.write_text(json.dumps({"b": [1, 2], "a": 1}))
    assert vc.config_hash(vc.load_config(p1)) == vc.config_hash(vc.load_config(p2))
    p2.write_text(json.dumps({"b": [1, 3], "a": 1}))
    assert vc.config_hash(vc.load_config(p1)) != vc.config_hash(vc.load_config(p2))


def test_repo_v8_config_is_loadable_and_weights_sum_to_one():
    cfg = vc.load_config(Path(__file__).resolve().parents[4] / "configs" / "v8_frozen.json")
    assert set(cfg["dimension_weights"]) == set(cfg["dimensions"])
    assert sum(cfg["dimension_weights"].values()) == pytest.approx(1.0)
    assert cfg["status"] in {"DRAFT", "FROZEN"}
    assert sum(1 for a in cfg["arms"].values() if a["primary"]) == 1


def test_repo_v8_config_embargo_covers_next_open_exit_of_longest_label():
    # T4 rule: an h-week label exits at the next open after t + h weeks, i.e. it settles h + 1 weeks later.
    cfg = vc.load_config(Path(__file__).resolve().parents[4] / "configs" / "v8_frozen.json")
    horizons = [int(t.rsplit("_", 1)[-1].rstrip("w")) for t in cfg["trained"]["targets"]]
    assert cfg["trained"]["label_embargo_weeks"] >= max(horizons) + 1


def test_repo_v8_config_uses_detailed_industry_and_user_caps():
    cfg = vc.load_config(Path(__file__).resolve().parents[4] / "configs" / "v8_frozen.json")
    assert cfg["industry_col"] == "industry_twse"
    ur = cfg["portfolios"]["user_rules"]
    assert ur["initial_capital_twd"] == 100000 and ur["industry_cap"] == 0.30
