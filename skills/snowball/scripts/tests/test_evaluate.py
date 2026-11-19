import numpy as np
import pandas as pd
import pytest

import evaluate as ev


def test_brier_and_skill_against_coin_flip():
    p = np.array([0.6, 0.6, 0.4, 0.4])
    o = np.array([1, 1, 0, 0])
    assert ev.brier(p, o) == pytest.approx(0.16)
    assert ev.brier_skill(p, o) == pytest.approx(1 - 0.16 / 0.25)


def test_murphy_decomposition_identity_with_constant_bins():
    p = np.array([0.3] * 10 + [0.7] * 10)
    o = np.array([0] * 7 + [1] * 3 + [1] * 6 + [0] * 4)
    rel, res, unc = ev.murphy(p, o, edges=(0.0, 0.5, 1.0))
    assert ev.brier(p, o) == pytest.approx(rel - res + unc)


def test_rank_ic_perfect_and_inverse():
    sig = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]})
    tgt = pd.DataFrame({"A": [0.1, 0.3], "B": [0.2, 0.2], "C": [0.3, 0.1]})
    ic = ev.rank_ic(sig, tgt, min_names=3)
    assert ic.iloc[0] == pytest.approx(1.0)
    assert ic.iloc[1] == pytest.approx(-1.0)


def test_ic_summary_t_stat():
    ic = pd.Series([0.05, 0.03, 0.04, 0.06])
    out = ev.ic_summary(ic)
    assert out["mean"] == pytest.approx(0.045)
    assert out["t"] > 0 and out["n"] == 4


def test_fit_logistic_recovers_positive_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-0.8 * x))).astype(float)
    a, k = ev.fit_logistic(x, y)
    assert k == pytest.approx(0.8, abs=0.15)
    assert abs(a) < 0.15


def test_long_short_spread_sign():
    sig = pd.DataFrame({c: [float(i)] for i, c in enumerate("ABCDEFGHIJ")})
    tgt = pd.DataFrame({c: [float(i) / 100] for i, c in enumerate("ABCDEFGHIJ")})
    spread = ev.long_short(sig, tgt, frac=0.2)
    assert spread.iloc[0] == pytest.approx(0.085 - 0.005)
