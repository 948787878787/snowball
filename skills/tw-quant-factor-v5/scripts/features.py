"""Pure feature functions for the v2 formula (section 9 of the research doc).

All panels are DataFrames indexed by week-end date with stock-id columns.
Every statistic at row t uses only rows <= t (time-series stats use rows < t).
"""

import numpy as np
import pandas as pd

Z_CLIP = 3.0


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("W-FRI").last()


def ts_zscore(df: pd.DataFrame, window: int = 52, min_periods: int = 26) -> pd.DataFrame:
    """Standardize each value by mean/std of the PREVIOUS `window` rows (no look-ahead)."""
    past = df.shift(1).rolling(window, min_periods=min_periods)
    z = (df - past.mean()) / past.std()
    return z.replace([np.inf, -np.inf], np.nan).clip(-Z_CLIP, Z_CLIP)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per row, clipped to +-3."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-Z_CLIP, Z_CLIP)


def rolling_beta(ret: pd.DataFrame, mkt: pd.Series, window: int = 52, min_periods: int = 26) -> pd.DataFrame:
    out = {}
    for col in ret.columns:
        r = ret[col]
        m = mkt.where(r.notna())
        roll = lambda s: s.rolling(window, min_periods=min_periods).mean()
        cov = roll(r * m) - roll(r) * roll(m)
        var = roll(m * m) - roll(m) ** 2
        out[col] = cov / var.replace(0.0, np.nan)
    return pd.DataFrame(out, index=ret.index)


def peer_mean_excl(alpha: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Equal-weight mean alpha of the OTHER members of each stock's peer group."""
    out = pd.DataFrame(np.nan, index=alpha.index, columns=alpha.columns)
    for members in groups.values():
        cols = [c for c in members if c in alpha.columns]
        block = alpha[cols]
        total = block.sum(axis=1, min_count=1)
        count = block.notna().sum(axis=1)
        for c in cols:
            own = block[c].fillna(0.0)
            n_other = count - block[c].notna().astype(int)
            out[c] = ((total - own) / n_other.replace(0, np.nan)).where(n_other > 0)
    return out


def leader_alpha(alpha: pd.DataFrame, turnover: pd.DataFrame, groups: dict):
    """Leader = member with the highest trailing turnover at that date.

    Returns (leader alpha broadcast to every member, follower flag L_i).
    """
    lead = pd.DataFrame(np.nan, index=alpha.index, columns=alpha.columns)
    follower = pd.DataFrame(np.nan, index=alpha.index, columns=alpha.columns)
    for members in groups.values():
        cols = [c for c in members if c in alpha.columns and c in turnover.columns]
        if len(cols) < 2:
            continue
        tv = turnover[cols].where(alpha[cols].notna())
        valid = tv.notna().any(axis=1).to_numpy()
        idx = tv.fillna(-np.inf).to_numpy().argmax(axis=1)
        a = alpha[cols].to_numpy()
        lead_vals = np.where(valid, a[np.arange(len(a)), idx], np.nan)
        for j, c in enumerate(cols):
            lead[c] = lead_vals
            follower[c] = np.where(valid, (idx != j).astype(float), np.nan)
    return lead, follower


def smooth_switch(dfund: pd.DataFrame, tau: float = 0.5) -> pd.DataFrame:
    return 1.0 / (1.0 + np.exp(-dfund / tau))


def peer_score_v2(zg, follower, zrel, s, v, theta=(1.0, 1.0, 1.0, 1.0)) -> pd.DataFrame:
    """F_peer = th1*L*z(G) - th2*s*Z + th3*(1-s)*min(Z,0) + th4*V   (doc section 9.3)."""
    t1, t2, t3, t4 = theta
    s_filled = s.fillna(0.5)
    term1 = (follower.fillna(0.0) * zg).fillna(0.0)
    term2 = -s_filled * zrel
    term3 = (1.0 - s_filled) * zrel.clip(upper=0.0)
    term4 = v.fillna(0.0)
    return t1 * term1 + t2 * term2 + t3 * term3 + t4 * term4


def peer_score_v1(lead, zdisp, dfund, val, theta=(1.0, 1.0, 1.0, 1.0)) -> pd.DataFrame:
    """Original draft formula, kept only to measure what the v2 corrections change."""
    t1, t2, t3, t4 = theta
    good = (dfund >= 0).astype(float).where(dfund.notna(), 1.0)
    return (t1 * lead.fillna(0.0)
            - t2 * good * zdisp
            + t3 * (1.0 - good) * np.sign(zdisp)
            + t4 * val.fillna(0.0))


def reversal_filter(s: pd.DataFrame, zabn: pd.DataFrame, lam: float = 0.3, threshold: float = 1.5) -> pd.DataFrame:
    active = (zabn.abs() > threshold).astype(float)
    return s - lam * active * zabn.clip(-Z_CLIP, Z_CLIP).fillna(0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
