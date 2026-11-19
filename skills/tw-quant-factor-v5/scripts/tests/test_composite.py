import numpy as np
import pandas as pd
import pytest

import composite as cp


def test_ic_weights_fall_back_to_prior_when_no_positive_icir():
    w = cp.ic_weights({"a": 0.6, "b": 0.4}, {"a": -0.1, "b": 0.0})
    assert w == pytest.approx({"a": 0.6, "b": 0.4})


def test_ic_weights_shrink_half_toward_prior_and_sum_to_one():
    w = cp.ic_weights({"a": 0.5, "b": 0.5}, {"a": 0.3, "b": -0.2})
    assert w["a"] == pytest.approx(0.75)
    assert w["b"] == pytest.approx(0.25)
    assert sum(w.values()) == pytest.approx(1.0)


def test_orthogonalize_removes_linear_exposure():
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(3, 40)))
    y = 2.0 * x + pd.DataFrame(rng.normal(size=(3, 40)))
    r = cp.orthogonalize(y, [x])
    for i in range(3):
        assert abs(np.corrcoef(r.iloc[i], x.iloc[i])[0, 1]) < 1e-8


def test_combine_masks_rows_without_required_signal():
    req = pd.DataFrame({"A": [1.0, np.nan]})
    pillars = {"k": pd.DataFrame({"A": [2.0, 2.0]})}
    out = cp.combine(pillars, {"k": 0.5}, req)
    assert out.iloc[0, 0] == pytest.approx(1.0)
    assert np.isnan(out.iloc[1, 0])
