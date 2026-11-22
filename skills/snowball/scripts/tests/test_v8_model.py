"""TDD tests for the v8 frozen Ridge arm: design matrix, sign constraints, embargo, persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import v8_composite as vc
import v8_model as vm

CFG = {
    "dimensions": {"a": {"x": 1, "y": -1}},
    "dimension_weights": {"a": 1.0},
    "zscore": {"by": "date", "winsor": 3.0, "min_names": 3},
    "trained": {"l2_alpha": 1.0, "targets": ["t4", "t12"], "label_embargo_weeks": 2,
                "extra_features": {"f_low_risk": 1, "e": 0}},
}


def _panel(n_dates: int = 30, n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-05", periods=n_dates, freq="W-FRI")
    df = pd.DataFrame({
        "date": np.repeat(dates, n), "ticker": np.tile([f"T{i}" for i in range(n)], n_dates),
        "x": rng.normal(size=n_dates * n), "y": rng.normal(size=n_dates * n),
        "f_low_risk": rng.normal(size=n_dates * n), "e": rng.normal(size=n_dates * n),
        "macro_stress": np.repeat(rng.normal(size=n_dates), n),
    })
    signal = df["x"] - df["y"] + 0.5 * df["f_low_risk"]
    df["t4"] = signal.groupby(df["date"]).rank(pct=True)
    df["t12"] = df["t4"]
    return vc.add_v8_zero(df, CFG)


def test_design_has_sign_adjusted_inputs_extras_and_interactions():
    X, names, bounds = vm.design_matrix(_panel(), CFG)
    assert {"sa_x", "sa_y", "f_low_risk", "e", "dim_a_x_stress"} == set(names)
    assert X.shape[1] == len(names) and not np.isnan(X).any()
    b = dict(zip(names, bounds))
    assert b["sa_x"] == (0.0, None) and b["f_low_risk"] == (0.0, None)
    assert b["e"] == (None, None) and b["dim_a_x_stress"] == (None, None)


def test_sign_adjusted_input_flips_negative_sign():
    df = _panel()
    X, names, _ = vm.design_matrix(df, CFG)
    z_y = vc.cs_zscore(df, "y", 3.0, 3).fillna(0.0).to_numpy()
    assert np.allclose(X[:, names.index("sa_y")], -z_y)


def test_fit_respects_bounds_and_recovers_positive_loadings():
    model = vm.fit(_panel(), CFG, train_end=pd.Timestamp("2024-12-31"))
    for tgt in ("t4", "t12"):
        coef = dict(zip(model["features"], model["coef"][tgt]))
        assert coef["sa_x"] > 0 and coef["sa_y"] > 0 and coef["f_low_risk"] >= 0


def test_fit_drops_rows_inside_label_embargo():
    df = _panel()
    last = df["date"].max()
    model = vm.fit(df, CFG, train_end=last)
    assert pd.Timestamp(model["train_last_date"]) == last - pd.Timedelta(weeks=2)


def test_predict_round_trips_through_json(tmp_path):
    df = _panel()
    model = vm.fit(df, CFG, train_end=df["date"].max())
    path = tmp_path / "m.json"
    vm.save_model(model, path)
    pred = vm.predict(df, CFG, vm.load_model(path))
    direct = vm.predict(df, CFG, model)
    assert np.allclose(pred, direct)
    ic = pd.Series(pred).groupby(df["date"].to_numpy()).corr(df["t4"].reset_index(drop=True)).mean()
    assert ic > 0.5


def test_predict_raises_when_feature_list_differs():
    df = _panel()
    model = vm.fit(df, CFG, train_end=df["date"].max())
    with pytest.raises(ValueError, match="feature"):
        vm.predict(df, CFG, {**model, "features": model["features"][:-1]})


def test_fit_design_matches_fit_on_precomputed_matrix():
    import v8_model as vm2
    df = _panel()
    end = df["date"].max()
    X, names, bounds = vm2.design_matrix(df, CFG)
    a = vm2.fit(df, CFG, train_end=end)
    b = vm2.fit_design(X, names, bounds, df, CFG, train_end=end)
    for t in CFG["trained"]["targets"]:
        assert np.allclose(a["coef"][t], b["coef"][t])
    assert a["train_last_date"] == b["train_last_date"]
