import pandas as pd
import pytest

import backtest_v3 as v3


def test_long_only_top_minus_universe_and_first_week_full_turnover():
    sig = pd.DataFrame({c: [float(i)] for i, c in enumerate("ABCDEFGHIJ")})
    tgt = pd.DataFrame({c: [float(i) / 100] for i, c in enumerate("ABCDEFGHIJ")})
    out = v3.long_only(sig, tgt, horizon=1, frac=0.2)
    gross_week = 0.085 - 0.045
    assert out["LO_excess_gross_ann"] == pytest.approx(gross_week * 52)
    assert out["LO_turnover"] == pytest.approx(1.0)
