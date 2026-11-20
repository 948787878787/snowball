"""Accuracy and signal-quality metrics: Brier / BSS / Murphy, rank IC, long-short spread, logistic fit."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr

COIN_FLIP_BRIER = 0.25


def brier(p, o) -> float:
    p, o = np.asarray(p, float), np.asarray(o, float)
    return float(np.mean((p - o) ** 2))


def brier_skill(p, o, ref: float = COIN_FLIP_BRIER) -> float:
    return 1.0 - brier(p, o) / ref


def murphy(p, o, edges=(0.0, 0.45, 0.5, 0.55, 0.6, 1.0)):
    """Return (reliability, resolution, uncertainty)."""
    p, o = np.asarray(p, float), np.asarray(o, float)
    n = len(p)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    obar = o.mean()
    rel = res = 0.0
    for k in np.unique(bins):
        m = bins == k
        rel += m.sum() * (p[m].mean() - o[m].mean()) ** 2
        res += m.sum() * (o[m].mean() - obar) ** 2
    return rel / n, res / n, obar * (1 - obar)


def rank_ic(signal: pd.DataFrame, target: pd.DataFrame, min_names: int = 10) -> pd.Series:
    """Per-date Spearman correlation between signal and target across stocks."""
    sig, tgt = signal.align(target, join="inner")
    values = {}
    for dt in sig.index:
        a, b = sig.loc[dt], tgt.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() >= min_names and a[m].nunique() > 1:
            values[dt] = spearmanr(a[m], b[m]).statistic
    return pd.Series(values, dtype=float)


def newey_west_t(x: pd.Series, lags: int = 4) -> float:
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(var / n))


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = len(ic)
    sd = ic.std()
    return {
        "mean": float(ic.mean()),
        "std": float(sd),
        "t": float(ic.mean() / sd * np.sqrt(n)) if sd > 0 else float("nan"),
        "t_nw": newey_west_t(ic),
        "hit": float((ic > 0).mean()),
        "n": n,
    }


def fit_logistic(x, y, l2: float = 1e-3):
    """Fit p = sigmoid(a + k*x) by maximum likelihood. Returns (a, k)."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def nll(w):
        z = w[0] + w[1] * x
        return np.sum(np.logaddexp(0.0, z) - y * z) / len(x) + l2 * w[1] ** 2

    res = minimize(nll, x0=np.zeros(2), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def long_short(signal: pd.DataFrame, target: pd.DataFrame, frac: float = 0.2) -> pd.Series:
    """Equal-weight top-frac minus bottom-frac target return per date."""
    sig, tgt = signal.align(target, join="inner")
    out = {}
    for dt in sig.index:
        a, b = sig.loc[dt], tgt.loc[dt]
        m = a.notna() & b.notna()
        k = int(np.floor(m.sum() * frac))
        if k < 1:
            continue
        order = a[m].sort_values().index
        out[dt] = b[order[-k:]].mean() - b[order[:k]].mean()
    return pd.Series(out, dtype=float)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    periods_per_year: float = 52.0,
) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (probability true SR > 0 after selection bias).

    Converts annualized `sharpe` to per-period SR, computes expected maximum null SR across `n_trials`,
    and returns P(SR_hat > SR_0) in [0, 1].
    """
    if not np.isfinite(sharpe) or n_obs < 4 or n_trials < 1:
        return float("nan")
    sr_per = float(sharpe) / np.sqrt(periods_per_year)
    if n_trials <= 1:
        sr_star = 0.0
    else:
        euler_gamma = 0.5772156649015329
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        # Null standard deviation of per-period SR estimates ~ 1 / sqrt(n_obs - 1)
        sr_std_null = 1.0 / np.sqrt(max(n_obs - 1, 1))
        sr_star = sr_std_null * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

    denom_var = max(1.0 - skew * sr_per + ((kurt - 1.0) / 4.0) * (sr_per ** 2), 1e-9)
    z_stat = (sr_per - sr_star) * np.sqrt(max(n_obs - 1, 1)) / np.sqrt(denom_var)
    return float(norm.cdf(z_stat))
