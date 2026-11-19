import numpy as np
import pandas as pd
import pytest

import features as ft

DATES = pd.date_range("2020-01-03", periods=6, freq="W-FRI")


def frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES[: len(next(iter(values.values())))])


class TestPeerScoreV2:
    def score(self, zg=0.0, lead=0.0, z=0.0, s=0.5, v=0.0):
        one = lambda x: pd.DataFrame({"A": [x]})
        return ft.peer_score_v2(one(zg), one(lead), one(z), one(s), one(v)).iloc[0, 0]

    def test_catch_up_is_positive_when_fundamentals_not_worse(self):
        assert self.score(z=-1.5, s=0.95) > 0

    def test_competitive_loss_is_negative_when_fundamentals_worse(self):
        assert self.score(z=-1.5, s=0.05) < 0

    def test_contagion_is_negative_when_stock_held_up_with_shared_factor(self):
        assert self.score(z=1.5, s=0.95) < 0

    def test_outperforming_with_worse_fundamentals_gets_no_reward(self):
        assert self.score(z=1.5, s=0.0) <= 0

    def test_leader_gap_ignored_when_stock_is_leader(self):
        assert self.score(zg=-2.0, lead=0.0) == 0
        assert self.score(zg=-2.0, lead=1.0) < 0

    def test_valuation_term_adds(self):
        assert self.score(v=1.0) > 0


def test_v1_draft_bug_rewards_worse_fundamentals_when_ahead():
    one = lambda x: pd.DataFrame({"A": [x]})
    v1 = ft.peer_score_v1(one(0.0), one(1.5), one(-1.0), one(0.0)).iloc[0, 0]
    assert v1 > 0  # documents the v1 sign error that v2 fixes


def test_ts_zscore_has_no_lookahead():
    base = frame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    changed = base.copy()
    changed.iloc[-1, 0] = 100.0
    z1 = ft.ts_zscore(base, window=4, min_periods=3)
    z2 = ft.ts_zscore(changed, window=4, min_periods=3)
    pd.testing.assert_frame_equal(z1.iloc[:-1], z2.iloc[:-1])


def test_ts_zscore_uses_only_past_values_for_stats():
    df = frame({"A": [1.0, 2.0, 3.0, 10.0]})
    z = ft.ts_zscore(df, window=3, min_periods=3)
    past = np.array([1.0, 2.0, 3.0])
    expected = (10.0 - past.mean()) / past.std(ddof=1)
    assert z.iloc[-1, 0] == pytest.approx(min(expected, 3.0))


def test_cs_zscore_rows_are_standardized_and_clipped():
    df = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [100.0]})
    z = ft.cs_zscore(df)
    assert z.max(axis=1).iloc[0] <= 3.0
    assert z.iloc[0, 0] < z.iloc[0, 1] < z.iloc[0, 2]


def test_peer_mean_excludes_self():
    alpha = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [6.0]})
    out = ft.peer_mean_excl(alpha, {"g": ("A", "B", "C")})
    assert out.loc[0, "A"] == pytest.approx(4.0)
    assert out.loc[0, "C"] == pytest.approx(1.5)


def test_leader_is_highest_turnover_and_flag_zero_for_leader():
    alpha = pd.DataFrame({"A": [0.1], "B": [-0.2], "C": [0.3]})
    turnover = pd.DataFrame({"A": [5.0], "B": [9.0], "C": [1.0]})
    lead_alpha, is_follower = ft.leader_alpha(alpha, turnover, {"g": ("A", "B", "C")})
    assert lead_alpha.loc[0, "A"] == pytest.approx(-0.2)
    assert is_follower.loc[0, "B"] == 0.0
    assert is_follower.loc[0, "A"] == 1.0


def test_smooth_switch_bounds_and_midpoint():
    s = ft.smooth_switch(pd.DataFrame({"A": [-10.0, 0.0, 10.0]}), tau=0.5)
    assert s.iloc[0, 0] < 0.01 and s.iloc[1, 0] == pytest.approx(0.5) and s.iloc[2, 0] > 0.99


def test_reversal_filter_only_acts_beyond_threshold():
    s = pd.DataFrame({"A": [1.0, 1.0]})
    zabn = pd.DataFrame({"A": [1.0, 2.0]})
    out = ft.reversal_filter(s, zabn, lam=0.3, threshold=1.5)
    assert out.iloc[0, 0] == pytest.approx(1.0)
    assert out.iloc[1, 0] == pytest.approx(1.0 - 0.6)


def test_rolling_beta_recovers_known_beta():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.02, 200))
    r = pd.DataFrame({"A": 1.5 * m + rng.normal(0, 0.001, 200)})
    beta = ft.rolling_beta(r, m, window=52, min_periods=26)
    assert beta.iloc[-1, 0] == pytest.approx(1.5, abs=0.05)
