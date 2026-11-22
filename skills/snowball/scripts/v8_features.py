"""Snowball v8 factor dimensions, all point-in-time at the weekly signal date (Friday close).

Dimensions (raw values; signs and weights live in the frozen v8 config, not here):
  chip    : 20-day foreign / investment-trust net buy over volume, 4-week margin-balance change over
            volume, 12-week change in foreign holding ratio
  risk    : 60-day idiosyncratic volatility, max daily return over 21 days, 20-day Amihud illiquidity
  macro   : macro transmission  sum_k beta_ik * shock_k  (v5 paper section 7.1 (A)), where beta are
            rolling 104-week loadings of weekly excess returns on USD/TWD, a monetary-identified US 10y
            yield change, and oil; shock_k is the last 4 weeks' factor move scaled by its 104-week vol
Quality lives in `tw_mops_financials`.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

FLOW_WINDOW = 20
MARGIN_WEEKS = 4
HOLD_WEEKS = 12
IDIO_WINDOW = 60
MAX_WINDOW = 21
AMIHUD_WINDOW = 20
BETA_WINDOW = 104
BETA_MIN_OBS = 52
SHOCK_WEEKS = 4
MONETARY_CORR_WINDOW = 13


def asof_weekly(daily: pd.DataFrame, weeks: pd.DatetimeIndex, max_gap_days: int = 7) -> pd.DataFrame:
    """Per column, the last valid value on or before each week date; NaN when that value is more than
    `max_gap_days` old. Per-column (not per-row) so a holiday in one market does not blank the others."""
    vals = daily.to_numpy(dtype=float)
    stamp = np.where(np.isfinite(vals), daily.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)[:, None], np.nan)
    filled_vals = pd.DataFrame(vals).ffill().to_numpy()
    filled_stamp = pd.DataFrame(stamp).ffill().to_numpy()
    idx = daily.index.searchsorted(weeks, side="right") - 1
    valid = idx >= 0
    take = np.where(valid, idx, 0)
    rows = filled_vals[take].copy()
    src = filled_stamp[take]
    week_ns = weeks.to_numpy(dtype="datetime64[ns]").astype(np.int64)[:, None]
    age_days = (week_ns - src) / 86_400e9
    stale = (~valid)[:, None] | ~np.isfinite(src) | (age_days > max_gap_days)
    rows[stale] = np.nan
    return pd.DataFrame(rows, index=weeks, columns=daily.columns)


def _tidy_to_wide(tidy: pd.DataFrame, value: str, id_to_ticker: Mapping[str, str]) -> pd.DataFrame:
    frame = tidy.assign(ticker=tidy["stock_id"].map(id_to_ticker)).dropna(subset=["ticker"])
    return frame.pivot_table(index="date", columns="ticker", values=value, aggfunc="last").sort_index()


def _stack(weekly: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide (date x ticker) frames sharing index/columns -> long frame; pandas-version independent."""
    first = next(iter(weekly.values()))
    dates = np.repeat(first.index.to_numpy(), first.shape[1])
    tickers = np.tile(first.columns.to_numpy(), first.shape[0])
    cols = {name: w.reindex(index=first.index, columns=first.columns).to_numpy(dtype=float).ravel()
            for name, w in weekly.items()}
    out = pd.DataFrame({"date": dates, "ticker": tickers, **cols})
    return out.dropna(how="all", subset=list(weekly)).reset_index(drop=True)


def chip_flow_features(inst: pd.DataFrame, volume: pd.DataFrame, weeks: pd.DatetimeIndex,
                       id_to_ticker: Mapping[str, str]) -> pd.DataFrame:
    """20-day institutional net buy as a share of 20-day traded volume (shares / shares)."""
    vol = volume.rolling(FLOW_WINDOW, min_periods=15).sum()
    out = {}
    for src, name in (("foreign_net", "chip_foreign_flow_20d"), ("trust_net", "chip_trust_flow_20d")):
        net = _tidy_to_wide(inst, src, id_to_ticker).reindex(columns=volume.columns)
        net = net.reindex(volume.index).fillna(0.0).where(volume.notna())
        flow = net.rolling(FLOW_WINDOW, min_periods=15).sum() / vol.where(vol > 0)
        out[name] = asof_weekly(flow, weeks)
    return _stack(out)


def chip_stock_features(margin: pd.DataFrame, hold: pd.DataFrame, volume: pd.DataFrame,
                        weeks: pd.DatetimeIndex, id_to_ticker: Mapping[str, str]) -> pd.DataFrame:
    """Margin-balance change (shares) over 20-day volume, and foreign-holding-ratio change (points)."""
    vol20 = asof_weekly(volume.rolling(FLOW_WINDOW, min_periods=15).sum(), weeks)
    m = asof_weekly(_tidy_to_wide(margin, "margin_lots", id_to_ticker), weeks).reindex(columns=volume.columns)
    h = asof_weekly(_tidy_to_wide(hold, "foreign_hold_pct", id_to_ticker), weeks).reindex(columns=volume.columns)
    return _stack({
        "chip_margin_chg_4w": (m - m.shift(MARGIN_WEEKS)) * 1000.0 / vol20.where(vol20 > 0),
        "chip_foreign_hold_chg_12w": h - h.shift(HOLD_WEEKS),
    })


def risk_features(close: pd.DataFrame, traded_value: pd.DataFrame, bench: pd.Series,
                  weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Idiosyncratic vol (market model, annualised), MAX lottery return, log Amihud illiquidity."""
    r = close.pct_change(fill_method=None)
    rm = bench.reindex(close.index).pct_change(fill_method=None)
    var_m = rm.rolling(IDIO_WINDOW, min_periods=40).var()
    cov = r.rolling(IDIO_WINDOW, min_periods=40).cov(rm)
    beta = cov.div(var_m, axis=0)
    resid_var = (r.rolling(IDIO_WINDOW, min_periods=40).var() - beta.pow(2).mul(var_m, axis=0)).clip(lower=0.0)
    amihud = (r.abs() / traded_value.where(traded_value > 0)).rolling(AMIHUD_WINDOW, min_periods=15).mean()
    return _stack({
        "risk_idio_vol_60d": asof_weekly(np.sqrt(resid_var * 252.0), weeks),
        "risk_max_21d": asof_weekly(r.rolling(MAX_WINDOW, min_periods=15).max(), weeks),
        "risk_amihud_20d": asof_weekly(np.log1p(amihud * 1e9), weeks),
    })


def monetary_rate_shock(d_yield: pd.Series, d_equity: pd.Series, window: int = MONETARY_CORR_WINDOW) -> pd.Series:
    """Keep the yield change only in weeks where trailing stock-bond co-movement is a discount-rate
    shock (corr(dy, equity return) < 0, i.e. yields up while equities fall)."""
    corr = d_yield.rolling(window, min_periods=window).corr(d_equity)
    return d_yield.where(corr < 0, 0.0)


def rolling_factor_betas(excess: pd.DataFrame, factors: pd.DataFrame, window: int = BETA_WINDOW,
                         min_obs: int = BETA_MIN_OBS) -> Dict[str, pd.DataFrame]:
    """Rolling multivariate OLS (with intercept) of each stock's weekly excess return on the factors,
    using only weeks where that stock has a return. Vectorised via rolling moment sums."""
    names = list(factors.columns)
    f = factors.reindex(excess.index).to_numpy(dtype=float)            # T x K
    y = excess.to_numpy(dtype=float)                                     # T x N
    mask = np.isfinite(y) & np.isfinite(f).all(axis=1)[:, None]
    y0, m = np.where(mask, y, 0.0), mask.astype(float)
    f0 = np.where(np.isfinite(f), f, 0.0)

    def roll(a: np.ndarray) -> np.ndarray:
        c = np.cumsum(a, axis=0)
        out = c.copy()
        out[window:] = c[window:] - c[:-window]
        return out

    n = roll(m)                                                          # T x N
    s_f = roll(m[:, :, None] * f0[:, None, :])                           # T x N x K
    s_y = roll(y0)
    s_ff = roll(m[:, :, None, None] * (f0[:, :, None] * f0[:, None, :])[:, None, :, :])
    s_fy = roll(y0[:, :, None] * f0[:, None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = n[:, :, None]
        cov_ff = s_ff / nn[..., None] - (s_f[..., :, None] * s_f[..., None, :]) / (nn[..., None] ** 2)
        cov_fy = s_fy / nn - s_f * (s_y[:, :, None] / nn ** 2)
    ok = n >= min_obs
    betas = np.full(s_f.shape, np.nan)
    idx = np.where(ok)
    if idx[0].size:
        sub_ff, sub_fy = cov_ff[idx], cov_fy[idx]
        good = np.abs(np.linalg.det(sub_ff)) > 1e-18
        sol = np.full(sub_fy.shape, np.nan)
        sol[good] = np.linalg.solve(sub_ff[good], sub_fy[good][..., None])[..., 0]
        betas[idx] = sol
    return {k: pd.DataFrame(betas[:, :, i], index=excess.index, columns=excess.columns) for i, k in enumerate(names)}


def macro_transmission_signal(betas: Dict[str, pd.DataFrame], factors: pd.DataFrame,
                              shock_weeks: int = SHOCK_WEEKS, window: int = BETA_WINDOW) -> pd.DataFrame:
    """sum_k beta_ik * (last `shock_weeks` factor change / (rolling weekly vol * sqrt(shock_weeks)))."""
    total = None
    for k, beta in betas.items():
        fk = factors[k].reindex(beta.index)
        scale = fk.rolling(window, min_periods=BETA_MIN_OBS).std() * np.sqrt(shock_weeks)
        shock = fk.rolling(shock_weeks, min_periods=shock_weeks).sum() / scale
        term = beta.mul(shock, axis=0)
        total = term if total is None else total + term
    return total


def macro_features(close_w: pd.DataFrame, bench_w: pd.Series, macro_w: pd.DataFrame) -> pd.DataFrame:
    """Weekly macro transmission signal. `macro_w` columns: usd (TWD per USD level), us10y (yield, %),
    oil (price level), spx (level); all sampled at the same Fridays as `close_w`."""
    ret = close_w.pct_change(fill_method=None)
    excess = ret.sub(bench_w.reindex(close_w.index).pct_change(fill_method=None), axis=0)
    d_y = macro_w["us10y"].diff()
    factors = pd.DataFrame({
        "usd": macro_w["usd"].pct_change(fill_method=None),
        "rate": monetary_rate_shock(d_y, macro_w["spx"].pct_change(fill_method=None)),
        "oil": macro_w["oil"].pct_change(fill_method=None),
    }, index=close_w.index)
    betas = rolling_factor_betas(excess, factors)
    signal = macro_transmission_signal(betas, factors)
    return _stack({"macro_transmission": signal})
