"""Macro-Micro Economic Integration Module (Point-in-Time Regime Conditioning & Microeconomic Moat).

Bridges Macroeconomics and Microeconomics grounded in peer-reviewed economics & finance literature:

1. Macroeconomics (Regime Conditioning — strictly lagged shift(1), zero look-ahead):
   - Term Structure / Yield Curve (`^TNX` 10Y - `^IRX` 3M): Estrella & Mishkin (1998 REStat);
     Harvey (1988 JFE); Lettau & Wachter (2007 JF, Equity Duration).
   - Credit Spreads & Financial Conditions (`HYG / IEF` + `^VIX`): Gilchrist & Zakrajsek (2012 AER);
     Adrian, Boyarchenko, & Giannone (2019 AER, Vulnerable Growth).
   - Global Dollar Liquidity Channel (`DX-Y.NYB`): Bruno & Shin (2015 RES); Obstfeld & Zhou (2022 BPEA).
   - Energy / Cost-Push Inflation Shock (`CL=F`): Kilian (2009 AER); Kanzig (2021 AER).

2. Microeconomics (Firm-Level Industrial Organization & Neoclassical q-Theory):
   - Lerner Pricing Power / Excess Markup (`GrossMargin - peer_median`): Lerner (1934 RES);
     De Loecker, Eeckhout, & Unger (2020 QJE); Peress (2010 JF).
   - Capital-Allocation Discipline & Profitability (`ROE`, `OpMargin`, `FCF/EV`, low leverage &
     liquidity buffer `CurrentRatio`): Novy-Marx (2013 JFE); Cooper, Gulen, & Schill (2008 JF);
     Hou, Xue, & Zhang (2015 RFS q-factor); Asness, Frazzini, & Pedersen (2019 RAST QMJ).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

MACRO_TICKERS = ("^TNX", "^IRX", "^VIX", "HYG", "IEF", "DX-Y.NYB", "CL=F")

BASELINE_V6_WEIGHTS = {
    "mom_qual": 0.20,
    "ind_mom": 0.15,
    "norm_val": 0.25,
    "quality": 0.20,
    "fund": 0.10,
    "low_risk": 0.10,
}

# Strict institutional risk bounds for dynamic macro-conditioned weights (all sum to 1.00 after normalization)
WEIGHT_BOUNDS = {
    "mom_qual": (0.10, 0.28),
    "ind_mom": (0.08, 0.22),
    "norm_val": (0.18, 0.34),
    "quality": (0.16, 0.32),
    "fund": (0.05, 0.16),
    "low_risk": (0.06, 0.22),
}


def download_macro_series(period: str = "5y") -> pd.DataFrame:
    """Download weekly-compatible daily macro indicator closes from Yahoo Finance."""
    try:
        raw = yf.download(list(MACRO_TICKERS), period=period, progress=False, auto_adjust=True, threads=True)
        if raw is None or raw.empty or "Close" not in raw:
            return pd.DataFrame()
        closes = raw["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=MACRO_TICKERS[0])
        return closes.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def make_offline_macro_data(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic synthetic daily macro indicators for offline testing and demos."""
    rng = np.random.default_rng(101)
    n = len(index)
    t = np.linspace(0.0, 4.0 * np.pi, n)
    tnx = 4.0 + 0.6 * np.sin(t) + np.cumsum(rng.normal(0.0, 0.015, n))
    irx = 3.8 + 0.5 * np.sin(t - 0.4) + np.cumsum(rng.normal(0.0, 0.010, n))
    vix = np.clip(18.0 + 5.0 * np.cos(t) + rng.normal(0.0, 1.2, n), 10.0, 50.0)
    ief = 95.0 * np.cumprod(1.0 + rng.normal(0.0001, 0.004, n))
    hyg = 78.0 * np.cumprod(1.0 + rng.normal(0.00015, 0.006, n))
    dxy = 103.0 * np.cumprod(1.0 + rng.normal(0.00005, 0.003, n))
    oil = 78.0 * np.cumprod(1.0 + rng.normal(0.00005, 0.012, n))
    return pd.DataFrame(
        {
            "^TNX": np.clip(tnx, 0.5, 10.0),
            "^IRX": np.clip(irx, 0.1, 10.0),
            "^VIX": vix,
            "HYG": hyg,
            "IEF": ief,
            "DX-Y.NYB": dxy,
            "CL=F": oil,
        },
        index=index,
    )


def _rolling_z(s: pd.Series, window: int = 52, min_periods: int = 13) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    return ((s - mu) / sd).clip(-3.0, 3.0)


def compute_macro_regime_panel(macro_d: Optional[pd.DataFrame], weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute strictly lagged (shift(1)) weekly macroeconomic regime indices:

    - `macro_stress` in [-1.5, +1.5]: Positive = tightening credit spreads / high VIX / inverted curve / strong USD.
    - `rate_inflation` in [-1.5, +1.5]: Positive = surging 10Y yields & energy cost-push inflation.
    - `regime`: Human-readable economic regime label.
    """
    neutral = pd.DataFrame(
        {
            "macro_stress": 0.0,
            "rate_inflation": 0.0,
            "term_spread": np.nan,
            "vix": np.nan,
            "regime": "Neutral / Balanced",
        },
        index=weeks,
    )
    if macro_d is None or macro_d.empty:
        return neutral

    m_w = macro_d.resample("W-FRI").last().ffill().reindex(weeks).ffill()
    # Strictly lag all macro state variables by 1 week (point-in-time)
    m_lag = m_w.shift(1)

    stress_parts = []
    stress_weights = []

    if "^VIX" in m_lag.columns and m_lag["^VIX"].notna().sum() >= 14:
        stress_parts.append(_rolling_z(m_lag["^VIX"]).fillna(0.0))
        stress_weights.append(0.35)

    if "HYG" in m_lag.columns and "IEF" in m_lag.columns:
        ratio = np.log((m_lag["HYG"] / m_lag["IEF"].replace(0.0, np.nan)).replace(0.0, np.nan))
        if ratio.notna().sum() >= 14:
            # Falling HYG/IEF ratio = widening credit spreads = higher financial stress
            stress_parts.append(-_rolling_z(ratio.diff(13)).fillna(0.0))
            stress_weights.append(0.35)

    term_spread = pd.Series(np.nan, index=weeks)
    if "^TNX" in m_lag.columns and "^IRX" in m_lag.columns:
        term_spread = m_lag["^TNX"] - m_lag["^IRX"]
        if term_spread.notna().sum() >= 14:
            # Flattening / inverting curve increases recession stress
            stress_parts.append(-_rolling_z(term_spread).fillna(0.0))
            stress_weights.append(0.15)

    if "DX-Y.NYB" in m_lag.columns and m_lag["DX-Y.NYB"].notna().sum() >= 14:
        # Rapid USD surge tightens global financial conditions (Bruno & Shin 2015)
        stress_parts.append(_rolling_z(m_lag["DX-Y.NYB"].pct_change(13, fill_method=None)).fillna(0.0))
        stress_weights.append(0.15)

    if stress_parts:
        w_sum = sum(stress_weights)
        macro_stress = sum(p * w for p, w in zip(stress_parts, stress_weights)) / w_sum
        macro_stress = macro_stress.clip(-1.5, 1.5).fillna(0.0)
    else:
        macro_stress = pd.Series(0.0, index=weeks)

    rate_parts = []
    rate_weights = []
    if "^TNX" in m_lag.columns and m_lag["^TNX"].notna().sum() >= 14:
        rate_parts.append(_rolling_z(m_lag["^TNX"].diff(13)).fillna(0.0))
        rate_weights.append(0.67)
    if "CL=F" in m_lag.columns and m_lag["CL=F"].notna().sum() >= 14:
        oil_chg = np.log(m_lag["CL=F"].replace(0.0, np.nan)).diff(13)
        rate_parts.append(_rolling_z(oil_chg).fillna(0.0))
        rate_weights.append(0.33)

    if rate_parts:
        rw_sum = sum(rate_weights)
        rate_inflation = sum(p * w for p, w in zip(rate_parts, rate_weights)) / rw_sum
        rate_inflation = rate_inflation.clip(-1.5, 1.5).fillna(0.0)
    else:
        rate_inflation = pd.Series(0.0, index=weeks)

    def _label(st: float, rt: float) -> str:
        if st >= 0.50 and rt >= 0.50:
            return "Stagflation / Tightening (Defensive + Pricing Power)"
        if st >= 0.50:
            return "Risk-Off / Credit Stress (Flight to Quality & Low-Risk)"
        if rt >= 0.50:
            return "Rate-Hike / Inflation Shock (Short-Duration Value + Markup)"
        if st <= -0.35 and rt <= 0.25:
            return "Expansion / Easy Conditions (Momentum + Growth Tilt)"
        return "Neutral / Balanced"

    regimes = [_label(float(s), float(r)) for s, r in zip(macro_stress, rate_inflation)]
    return pd.DataFrame(
        {
            "macro_stress": macro_stress,
            "rate_inflation": rate_inflation,
            "term_spread": term_spread,
            "vix": m_lag["^VIX"] if "^VIX" in m_lag.columns else np.nan,
            "regime": regimes,
        },
        index=weeks,
    )


def regime_conditioned_weights(macro_stress: float, rate_inflation: float) -> Dict[str, float]:
    """Map lagged macro regime indices into smooth, bounded 6-pillar factor weights summing to 1.00.

    - Higher `macro_stress` -> rotates toward Micro Quality/Moat & Defensive Low-Risk; trims crash-prone Momentum.
    - Higher `rate_inflation` -> rotates toward Short-Duration Value & Pricing Power; trims long-duration Growth.
    """
    st = float(np.clip(macro_stress, -1.5, 1.5))
    rt = float(np.clip(rate_inflation, -1.5, 1.5))

    raw = {
        "mom_qual": BASELINE_V6_WEIGHTS["mom_qual"] - 0.05 * st - 0.03 * max(0.0, rt),
        "ind_mom":  BASELINE_V6_WEIGHTS["ind_mom"]  - 0.04 * st,
        "norm_val": BASELINE_V6_WEIGHTS["norm_val"] + 0.04 * rt - 0.01 * max(0.0, st),
        "quality":  BASELINE_V6_WEIGHTS["quality"]  + 0.05 * st + 0.03 * max(0.0, rt),
        "fund":     BASELINE_V6_WEIGHTS["fund"]     - 0.02 * max(0.0, st) - 0.03 * max(0.0, rt),
        "low_risk": BASELINE_V6_WEIGHTS["low_risk"] + 0.05 * st - 0.01 * min(0.0, st),
    }

    clipped = {k: float(np.clip(v, WEIGHT_BOUNDS[k][0], WEIGHT_BOUNDS[k][1])) for k, v in raw.items()}
    total = sum(clipped.values())
    return {k: v / total for k, v in clipped.items()}


def compute_micro_lerner_markup(gross_margin: pd.Series, groups: dict) -> pd.Series:
    """Microeconomic Lerner Index excess markup proxy: firm gross margin minus peer-group median.

    Measures relative pricing power & demand inelasticity inside each industry
    (Lerner 1934; De Loecker, Eeckhout, & Unger 2020 QJE).
    """
    gm = gross_margin.clip(-0.20, 0.95)
    out = pd.Series(np.nan, index=gm.index, dtype=float)
    for members in groups.values():
        cols = [c for c in members if c in gm.index]
        if not cols:
            continue
        sub = gm.loc[cols]
        med = sub.median()
        if pd.notna(med):
            out.loc[cols] = sub - med
    # Fallback to cross-sectional median for ungrouped custom tickers
    missing = out.isna() & gm.notna()
    if missing.any():
        out.loc[missing] = gm.loc[missing] - gm.median()
    return out
