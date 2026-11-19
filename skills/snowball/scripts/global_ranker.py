#!/usr/bin/env python3
"""Universal Global Equity Multi-Factor Ranker & Point-in-Time Backtester (v6 & v5).

Works across ANY Yahoo Finance equity market (US 1,000, TW 1,000, EU 260, JP 260, HK 260,
GLOBAL 2,780+ stocks across 13 standardized industries), custom JSON universe files,
or arbitrary comma-separated tickers.

Default v6 6-Pillar Empirical Formula (Peer-Reviewed Asset Pricing Architecture):
1. Momentum Quality (F_MomQual, 20%):
   - Volatility-Scaled Residual Momentum (Blitz, Huij, & Martens 2011 JEF; Barroso & Santa-Clara 2015 JFE):
     26w skipping last 4w cumulative Beta-adjusted alpha divided by trailing residual volatility.
   - Frog-in-the-Pan Continuous Information (Da, Gurun, & Warachka 2014 RFS):
     ID = sgn(PRET) * (%neg_weeks - %pos_weeks); rewards smooth, persistent trends over lottery jumps.
2. Full-Group Industry Momentum (F_IndMom_Full_12w, 15%):
   - Moskowitz & Grinblatt (1999 JF); Ehsani & Linnainmaa (2022 JF).
3. Multi-Metric Capital-Structure-Neutral Value (F_NormVal, 25%):
   - Non-financials: within-industry 0.5*z(min(E/P, 15%)) + 0.5*z(min(EBITDA/EV, 25%))
     (Loughran & Wellman 2011 JFQA; strictly trailing P/E, no forwardPE fallback for loss firms).
   - Financials: within-industry z(1/PBR) (Bhojraj & Lee 2002 JAR).
   - Payout component: + 0.5*z(min(DividendYield, 8.0%)) (Boudoukh et al. 2007 JF).
4. Quality & Cash-Flow Profitability (F_Quality, 20%):
   - Novy-Marx (2013 JFE), Fama & French (2015 JFE), Ball et al. (2016 JFE), Asness, Frazzini, & Pedersen (2019 RAST):
     within-industry ROE, Operating Margin, Gross Margin + FCF Yield - Non-Financial Debt/Equity.
5. Fundamental Growth (F_Fund, 10%):
   - Revenue Growth YoY + Earnings Growth YoY (Ball & Brown 1968 JAR; Bernard & Thomas 1989 JAR).
6. Defensive Low-Risk (F_LowRisk, 10%):
   - Betting Against Beta (Frazzini & Pedersen 2014 JFE) + Low Idiosyncratic Volatility (Ang et al. 2006 JF).

Audit & Anti-Look-Ahead Guarantees:
- Next-trading-day Open execution (Monday open fill after Friday close signal, eliminating same-bar bias).
- Automatic USD FX conversion in multi-currency / GLOBAL runs so Beta vs ACWI is currency-consistent.
- Optional point-in-time historical membership mask (`--membership-csv`) for survivorship-free tests.

Usage:
    python global_ranker.py --market US --top 20
    python global_ranker.py --market US --model v5 --top 20
    python global_ranker.py --market US --industry semiconductors_hardware
    python global_ranker.py --market GLOBAL --backtest --skip-fundamentals
    python global_ranker.py --universe-file ./my_extra_universe.json
    python global_ranker.py --market US --offline-demo
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate as ev
import features as ft
from backtest_v5 import EY_CAP, NW_LAGS, YIELD_CAP_PCT, _group_mean_incl, _group_zscore
from global_universe import (
    GLOBAL_PB_GROUPS,
    MARKET_CONFIG,
    get_market_groups,
    get_market_tickers,
    group_map_for,
    list_markets,
    load_universe_file,
)

BETA_WINDOW = 52
BETA_MIN = 26
EBITDA_EV_CAP = 0.25
FCF_YIELD_CAP = 0.15

SUFFIX_CURRENCY = {
    ".T": "JPY",
    ".TW": "TWD",
    ".TWO": "TWD",
    ".HK": "HKD",
    ".L": "GBP",
    ".SW": "CHF",
    ".CO": "DKK",
    ".ST": "SEK",
    ".OL": "NOK",
    ".KS": "KRW",
    ".KQ": "KRW",
    ".PA": "EUR",
    ".DE": "EUR",
    ".AS": "EUR",
    ".MI": "EUR",
    ".MC": "EUR",
    ".BR": "EUR",
    ".HE": "EUR",
    ".IR": "EUR",
    ".LS": "EUR",
    ".VI": "EUR",
}

FX_TO_USD_TICKERS = {
    "EUR": ("EURUSD=X", True),
    "GBP": ("GBPUSD=X", True),
    "JPY": ("JPY=X", False),
    "TWD": ("TWD=X", False),
    "HKD": ("HKD=X", False),
    "CHF": ("CHF=X", False),
    "DKK": ("DKK=X", False),
    "SEK": ("SEK=X", False),
    "NOK": ("NOK=X", False),
    "KRW": ("KRW=X", False),
}


def infer_ticker_currency(ticker: str) -> str:
    """Infer trading currency from Yahoo Finance exchange suffix (defaults to USD)."""
    sym = ticker.upper().strip()
    for suffix, curr in SUFFIX_CURRENCY.items():
        if sym.endswith(suffix):
            return curr
    return "USD"


def convert_prices_to_usd(
    close_d: pd.DataFrame,
    open_d: Optional[pd.DataFrame] = None,
    fx_series_map: Optional[Dict[str, Tuple[pd.Series, bool]]] = None,
    period: str = "5y",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Convert non-USD daily stock prices into USD using historical FX series."""
    needed_ccys = sorted({
        infer_ticker_currency(c) for c in close_d.columns if infer_ticker_currency(c) != "USD"
    })
    if not needed_ccys:
        return close_d, open_d

    fx_map: Dict[str, Tuple[pd.Series, bool]] = dict(fx_series_map or {})
    missing_ccys = [c for c in needed_ccys if c not in fx_map and c in FX_TO_USD_TICKERS]
    if missing_ccys:
        fx_syms = [FX_TO_USD_TICKERS[c][0] for c in missing_ccys]
        raw = yf.download(fx_syms, period=period, progress=False, auto_adjust=True, threads=True)
        if raw is not None and not raw.empty and "Close" in raw:
            fx_close = raw["Close"]
            if isinstance(fx_close, pd.Series):
                fx_close = fx_close.to_frame(name=fx_syms[0])
            for c in missing_ccys:
                sym, is_usd_per_local = FX_TO_USD_TICKERS[c]
                if sym in fx_close.columns and not fx_close[sym].dropna().empty:
                    fx_map[c] = (fx_close[sym].dropna(), is_usd_per_local)

    close_usd = close_d.copy()
    open_usd = open_d.copy() if open_d is not None else None
    for col in close_d.columns:
        ccy = infer_ticker_currency(col)
        if ccy == "USD" or ccy not in fx_map:
            continue
        fx_s, is_usd_per_local = fx_map[ccy]
        aligned = fx_s.reindex(close_usd.index).ffill().bfill()
        if is_usd_per_local:
            close_usd[col] = close_usd[col] * aligned
            if open_usd is not None and col in open_usd.columns:
                open_usd[col] = open_usd[col] * aligned.reindex(open_usd.index).ffill().bfill()
        else:
            safe_fx = aligned.replace(0.0, np.nan)
            close_usd[col] = close_usd[col] / safe_fx
            if open_usd is not None and col in open_usd.columns:
                open_usd[col] = open_usd[col] / safe_fx.reindex(open_usd.index).ffill().bfill()
    return close_usd, open_usd


def download_global_prices(
    tickers: list,
    benchmark: str,
    period: str = "5y",
    batch_size: int = 80,
) -> tuple:
    """Download dividend-adjusted daily Close and Open prices in batches from Yahoo Finance.

    Returns (close_d, open_d, bench_d).
    """
    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        raise ValueError("No tickers provided.")

    bench_raw = yf.download([benchmark], period=period, progress=False, auto_adjust=True, threads=False)
    if bench_raw is None or bench_raw.empty or "Close" not in bench_raw:
        raise RuntimeError(f"Benchmark '{benchmark}' returned no price history.")
    bench_close = bench_raw["Close"]
    if isinstance(bench_close, pd.DataFrame):
        bench_close = bench_close.iloc[:, 0]
    bench_series = bench_close.dropna().rename(benchmark)
    if bench_series.empty:
        raise RuntimeError(f"Benchmark '{benchmark}' returned no price history.")

    close_frames: List[pd.DataFrame] = []
    open_frames: List[pd.DataFrame] = []
    for start in range(0, len(unique_tickers), batch_size):
        chunk = unique_tickers[start:start + batch_size]
        raw = yf.download(chunk, period=period, progress=False, auto_adjust=True, threads=True)
        if raw is None or raw.empty or "Close" not in raw:
            continue
        closes = raw["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=chunk[0])
        close_frames.append(closes)
        if "Open" in raw:
            opens = raw["Open"]
            if isinstance(opens, pd.Series):
                opens = opens.to_frame(name=chunk[0])
            open_frames.append(opens)

    if not close_frames:
        raise RuntimeError(f"Unable to fetch Yahoo Finance prices for {len(unique_tickers)} symbols.")

    all_closes = pd.concat(close_frames, axis=1)
    all_closes = all_closes.loc[:, ~all_closes.columns.duplicated()]
    stock_cols = [t for t in unique_tickers if t in all_closes.columns and all_closes[t].dropna().shape[0] >= 30]
    if not stock_cols:
        raise RuntimeError("No stock symbols returned sufficient price history.")

    if open_frames:
        all_opens = pd.concat(open_frames, axis=1)
        all_opens = all_opens.loc[:, ~all_opens.columns.duplicated()]
        open_df = all_opens.reindex(index=all_closes.index, columns=stock_cols)
    else:
        open_df = all_closes[stock_cols].copy()

    return all_closes[stock_cols], open_df, bench_series


def parse_yahoo_dividend_yield_pct(info: dict) -> float:
    """Parse trailing annual dividend yield in percentage points (e.g., 0.42 -> 0.42%) safely."""
    rate = info.get("trailingAnnualDividendRate")
    px = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
    if rate is not None and px is not None:
        try:
            r_val, p_val = float(rate), float(px)
            if r_val >= 0.0 and p_val > 0.0:
                return max(0.0, (r_val / p_val) * 100.0)
        except (TypeError, ValueError):
            pass

    ty = info.get("trailingAnnualDividendYield")
    if ty is not None:
        try:
            ty_val = float(ty)
            if 0.0 <= ty_val < 0.50:
                return ty_val * 100.0
        except (TypeError, ValueError):
            pass

    dy = info.get("dividendYield")
    if dy is not None:
        try:
            dy_val = float(dy)
            if dy_val >= 0.0:
                return dy_val
        except (TypeError, ValueError):
            pass
    return 0.0


def parse_yahoo_trailing_pe(info: dict) -> float:
    """Return strictly positive trailing P/E; never fall back to forwardPE for loss-making firms."""
    eps = info.get("trailingEps")
    if eps is not None:
        try:
            if float(eps) <= 0.0:
                return np.nan
        except (TypeError, ValueError):
            pass
    per = info.get("trailingPE")
    if per is not None:
        try:
            per_val = float(per)
            if per_val > 0.0:
                return per_val
        except (TypeError, ValueError):
            pass
    return np.nan


def _safe_float(val, positive_only: bool = False) -> float:
    if val is None:
        return np.nan
    try:
        x = float(val)
        if not np.isfinite(x):
            return np.nan
        if positive_only and x <= 0.0:
            return np.nan
        return x
    except (TypeError, ValueError):
        return np.nan


def _fetch_one_fundamental(sym: str) -> dict:
    try:
        info = yf.Ticker(sym).info or {}
    except Exception:
        info = {}
    dy_pct = parse_yahoo_dividend_yield_pct(info)
    per = parse_yahoo_trailing_pe(info)
    pbr = _safe_float(info.get("priceToBook"), positive_only=True)
    ev_ebitda = _safe_float(info.get("enterpriseToEbitda"), positive_only=True)
    rev_g = _safe_float(info.get("revenueGrowth"))
    earn_g = _safe_float(info.get("earningsGrowth"))
    roe = _safe_float(info.get("returnOnEquity"))
    op_margin = _safe_float(info.get("operatingMargins"))
    gross_margin = _safe_float(info.get("grossMargins"))
    d2e = _safe_float(info.get("debtToEquity"))
    if np.isfinite(d2e) and d2e < 0.0:
        d2e = np.nan

    fcf = _safe_float(info.get("freeCashflow"))
    ev_val = _safe_float(info.get("enterpriseValue"), positive_only=True)
    mcap = _safe_float(info.get("marketCap"), positive_only=True)
    denom = ev_val if np.isfinite(ev_val) else mcap
    fcf_yield = (fcf / denom) if (np.isfinite(fcf) and np.isfinite(denom) and denom > 0.0) else np.nan

    return {
        "ticker": sym,
        "name": info.get("shortName") or sym,
        "sector": info.get("sector") or "",
        "currency": info.get("currency") or infer_ticker_currency(sym),
        "dividend_yield": max(0.0, dy_pct),
        "PER": per,
        "PBR": pbr,
        "ev_ebitda": ev_ebitda,
        "rev_growth": rev_g,
        "earnings_growth": earn_g,
        "roe": roe,
        "op_margin": op_margin,
        "gross_margin": gross_margin,
        "fcf_yield": fcf_yield,
        "debt_to_equity": d2e,
    }


def fetch_global_fundamentals(tickers: list, max_workers: int = 8) -> pd.DataFrame:
    """Fetch point-in-time snapshot fundamentals concurrently from Yahoo Finance."""
    cols = [
        "name", "sector", "currency", "dividend_yield", "PER", "PBR", "ev_ebitda",
        "rev_growth", "earnings_growth", "roe", "op_margin", "gross_margin",
        "fcf_yield", "debt_to_equity",
    ]
    if not tickers:
        return pd.DataFrame(columns=cols)
    workers = max(1, min(max_workers, len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_fetch_one_fundamental, tickers))
    return pd.DataFrame(rows).set_index("ticker")


def make_neutral_fundamentals(tickers: list) -> pd.DataFrame:
    """Neutral placeholder fundamentals when running in fast price-only mode."""
    return pd.DataFrame({
        "ticker": tickers,
        "name": tickers,
        "sector": "",
        "currency": [infer_ticker_currency(t) for t in tickers],
        "dividend_yield": 0.0,
        "PER": np.nan,
        "PBR": np.nan,
        "ev_ebitda": np.nan,
        "rev_growth": np.nan,
        "earnings_growth": np.nan,
        "roe": np.nan,
        "op_margin": np.nan,
        "gross_margin": np.nan,
        "fcf_yield": np.nan,
        "debt_to_equity": np.nan,
    }).set_index("ticker")


def compute_next_open_weekly(open_d: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """For each weekly Friday signal timestamp `dt`, return the first trading day's Open strictly after `dt`."""
    open_sorted = open_d.sort_index()
    idx = open_sorted.index
    rows = []
    for dt in weeks:
        pos = int(idx.searchsorted(dt, side="right"))
        if pos < len(open_sorted):
            rows.append(open_sorted.iloc[pos])
        else:
            rows.append(pd.Series(np.nan, index=open_sorted.columns))
    return pd.DataFrame(rows, index=weeks, columns=open_sorted.columns)


def compute_global_price_factors(
    close_d: pd.DataFrame,
    bench_d: pd.Series,
    groups: dict,
    open_d: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute weekly point-in-time v5 & v6 price factors and next-trading-day Open prices."""
    close_w = close_d.resample("W-FRI").last().ffill()
    bench_w = bench_d.resample("W-FRI").last().ffill().reindex(close_w.index)
    next_open_w = (
        compute_next_open_weekly(open_d, close_w.index)
        if open_d is not None
        else close_w.shift(-1)
    )

    ret_1w = close_w.pct_change(fill_method=None)
    mkt_1w = bench_w.pct_change(fill_method=None)

    beta = ft.rolling_beta(ret_1w, mkt_1w, BETA_WINDOW, BETA_MIN)
    beta_lag = beta.shift(1)
    alpha = ret_1w - beta_lag.mul(mkt_1w, axis=0)
    target_4w = sum(alpha.shift(-k) for k in range(1, 5))

    # Trailing idiosyncratic volatility (26w)
    ivol = alpha.rolling(26, min_periods=13).std()

    # 1a. Raw Residual Momentum (v5): 26w skipping last 4w cumulative alpha
    res_mom_raw = alpha.shift(4).rolling(22, min_periods=12).sum()
    f_res_mom = ft.cs_zscore(res_mom_raw)

    # 1b. Volatility-Scaled Residual Momentum (Blitz et al. 2011 JEF; Barroso & Santa-Clara 2015 JFE)
    res_vol = alpha.shift(4).rolling(22, min_periods=12).std().replace(0.0, np.nan)
    res_mom_vol_raw = res_mom_raw / (res_vol * np.sqrt(22.0))
    f_res_mom_vol = ft.cs_zscore(res_mom_vol_raw)

    # 1c. Frog-in-the-Pan Continuous Information (Da, Gurun, & Warachka 2014 RFS)
    # ID = sgn(PRET_26_4w) * (%neg_weeks - %pos_weeks); negative ID = smooth persistent trend
    ret_shifted = ret_1w.shift(4)
    pret_26_4 = (1.0 + ret_shifted).rolling(22, min_periods=12).apply(np.prod, raw=True) - 1.0
    pos_frac = (ret_shifted > 0.0).astype(float).rolling(22, min_periods=12).mean()
    neg_frac = (ret_shifted < 0.0).astype(float).rolling(22, min_periods=12).mean()
    info_discreteness = np.sign(pret_26_4) * (neg_frac - pos_frac)
    f_fip = ft.cs_zscore(-info_discreteness)

    # Combined Momentum Quality (75% Vol-Scaled ResMom + 25% Frog-in-the-Pan Smoothness)
    f_mom_qual = ft.cs_zscore(0.75 * f_res_mom_vol.fillna(0.0) + 0.25 * f_fip.fillna(0.0)).where(f_res_mom_vol.notna())

    # 2. Full-Group Industry Momentum (12w inclusive group mean alpha, Moskowitz & Grinblatt 1999 JF)
    alpha_12w = alpha.rolling(12, min_periods=6).sum()
    f_ind_mom = ft.cs_zscore(_group_mean_incl(alpha_12w, groups))

    # 3. Defensive Low-Risk: Betting Against Beta (Frazzini & Pedersen 2014 JFE) + Low IVOL (Ang et al. 2006 JF)
    f_low_vol = ft.cs_zscore(-ivol)
    f_bab = ft.cs_zscore(-beta_lag)
    f_low_risk = ft.cs_zscore(0.5 * f_bab.fillna(0.0) + 0.5 * f_low_vol.fillna(0.0)).where(ivol.notna())

    return {
        "close_w": close_w,
        "next_open_w": next_open_w,
        "ret_1w": ret_1w,
        "mkt_1w": mkt_1w,
        "beta": beta_lag,
        "ivol": ivol,
        "alpha": alpha,
        "target_4w": target_4w,
        "f_res_mom": f_res_mom,
        "f_res_mom_vol": f_res_mom_vol,
        "f_fip": f_fip,
        "f_mom_qual": f_mom_qual,
        "f_ind_mom": f_ind_mom,
        "f_low_vol": f_low_vol,
        "f_low_risk": f_low_risk,
    }


def _cs_z_series(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not (sd and np.isfinite(sd) and sd > 0.0):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).clip(-3.0, 3.0).fillna(0.0)


def _group_z_series(s: pd.Series, groups: dict) -> pd.Series:
    df = s.to_frame().T
    gz = _group_zscore(df, groups).iloc[0]
    if gz.isna().all():
        return _cs_z_series(s)
    return gz.fillna(0.0)


def score_latest_global(
    price_factors: dict,
    fundamentals: pd.DataFrame,
    groups: dict,
    model: str = "v6",
) -> pd.DataFrame:
    """Score latest global cross-section using v6 (6-pillar) or v5 (4-pillar) empirical formula."""
    dt = price_factors["f_res_mom"].index[-1]
    tickers = list(price_factors["close_w"].columns)
    gmap = group_map_for(groups)

    f_res = price_factors["f_res_mom"].loc[dt].reindex(tickers).fillna(0.0)
    f_mom_qual = price_factors["f_mom_qual"].loc[dt].reindex(tickers).fillna(0.0)
    f_fip = price_factors["f_fip"].loc[dt].reindex(tickers).fillna(0.0)
    f_ind = price_factors["f_ind_mom"].loc[dt].reindex(tickers).fillna(0.0)
    f_low_risk = price_factors["f_low_risk"].loc[dt].reindex(tickers).fillna(0.0)

    fund = fundamentals.reindex(tickers)
    for col in ("ev_ebitda", "earnings_growth", "roe", "op_margin", "gross_margin", "fcf_yield", "debt_to_equity"):
        if col not in fund.columns:
            fund[col] = np.nan

    # Pillar: Multi-Metric Capital-Structure-Neutral Value (E/P + EBITDA/EV for non-fin; B/P for fin; + 0.5*Capped Yield)
    ey = (1.0 / fund["PER"].where(fund["PER"] > 0)).clip(upper=EY_CAP).fillna(0.0)
    bp = 1.0 / fund["PBR"].where(fund["PBR"] > 0)
    ebitda_ev = (1.0 / fund["ev_ebitda"].where(fund["ev_ebitda"] > 0)).clip(upper=EBITDA_EV_CAP)
    yld_cap = fund["dividend_yield"].fillna(0.0).clip(upper=YIELD_CAP_PCT)

    ey_ind = _group_z_series(ey, groups)
    bp_ind = _group_z_series(bp, groups)
    ev_ind = _group_z_series(ebitda_ev, groups)
    has_ev = ebitda_ev.notna()

    val_core = pd.Series(
        {
            t: (
                bp_ind[t]
                if gmap.get(t) in GLOBAL_PB_GROUPS
                else (0.5 * ey_ind[t] + 0.5 * ev_ind[t] if (model == "v6" and bool(has_ev[t])) else ey_ind[t])
            )
            for t in tickers
        },
        dtype=float,
    )
    val_yld = _cs_z_series(yld_cap)
    f_norm_val = _cs_z_series(val_core + 0.5 * val_yld)

    # Pillar: Quality & Cash-Flow Profitability (Novy-Marx 2013, Fama-French 2015, Ball et al. 2016, AFP 2019 QMJ)
    z_roe = _group_z_series(fund["roe"].clip(-0.5, 1.0), groups)
    z_opm = _group_z_series(fund["op_margin"].clip(-0.5, 0.8), groups)
    z_gm = _group_z_series(fund["gross_margin"].clip(-0.2, 0.95), groups)
    z_fcf = _cs_z_series(fund["fcf_yield"].clip(-FCF_YIELD_CAP, FCF_YIELD_CAP))
    non_fin_d2e = pd.Series(
        {t: (np.nan if gmap.get(t) in GLOBAL_PB_GROUPS else fund.loc[t, "debt_to_equity"]) for t in tickers},
        dtype=float,
    ).clip(0.0, 500.0)
    z_lev = _cs_z_series(non_fin_d2e)
    f_quality = _cs_z_series(0.40 * z_roe + 0.30 * z_opm + 0.30 * z_gm + 0.25 * z_fcf - 0.25 * z_lev)

    # Pillar: Fundamental Growth (Revenue YoY + Earnings YoY)
    z_rev = _cs_z_series(fund["rev_growth"].clip(-0.5, 1.5))
    z_earn = _cs_z_series(fund["earnings_growth"].clip(-0.5, 1.5))
    has_earn = fund["earnings_growth"].notna().any()
    f_fin = _cs_z_series(0.60 * z_rev + 0.40 * z_earn) if (model == "v6" and has_earn) else z_rev

    # Composite Scores
    raw_v5 = 0.5 * f_res + 0.5 * f_ind + 1.0 * z_rev + 1.5 * f_norm_val
    v5_score = _cs_z_series(raw_v5)

    raw_v6 = (
        0.20 * f_mom_qual
        + 0.15 * f_ind
        + 0.25 * f_norm_val
        + 0.20 * f_quality
        + 0.10 * f_fin
        + 0.10 * f_low_risk
    )
    v6_score = _cs_z_series(raw_v6)

    score_col = "v5_score" if model.lower() == "v5" else "v6_score"
    active_score = v5_score if model.lower() == "v5" else v6_score

    out = pd.DataFrame({
        "ticker": tickers,
        "name": fund["name"].fillna(pd.Series(tickers, index=tickers)).values,
        "group": [gmap.get(t, "custom") for t in tickers],
        "close": price_factors["close_w"].loc[dt, tickers].values,
        "yield_%": fund["dividend_yield"].fillna(0.0).values,
        "PER": fund["PER"].values,
        "rev_yoy_%": (fund["rev_growth"] * 100.0).values,
        "mom_qual": f_mom_qual.values,
        "fip": f_fip.values,
        "ind_mom": f_ind.values,
        "norm_val": f_norm_val.values,
        "quality": f_quality.values,
        "low_risk": f_low_risk.values,
        "v5_score": v5_score.values,
        "v6_score": v6_score.values,
    }).sort_values(score_col, ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def backtest_global_staggered(
    price_factors: dict,
    round_trip_cost: float = 0.0020,
    buy_frac: float = 0.20,
    membership_mask: Optional[pd.DataFrame] = None,
    model: str = "v6",
) -> dict:
    """4-cohort staggered historical backtest with realistic Next-Trading-Day Open execution.

    Pure-price point-in-time signals:
    - v6: 0.45 * Vol-Scaled ResMom + 0.15 * Frog-in-the-Pan + 0.25 * IndMom + 0.15 * LowRisk (BAB+IVOL)
    - v5: 0.50 * ResMom + 0.50 * IndMom
    """
    if model.lower() == "v5":
        sig = ft.cs_zscore(
            price_factors["f_res_mom"].fillna(0.0) + price_factors["f_ind_mom"].fillna(0.0)
        ).where(price_factors["f_res_mom"].notna())
    else:
        sig = ft.cs_zscore(
            0.45 * price_factors["f_res_mom_vol"].fillna(0.0)
            + 0.15 * price_factors["f_fip"].fillna(0.0)
            + 0.25 * price_factors["f_ind_mom"].fillna(0.0)
            + 0.15 * price_factors["f_low_risk"].fillna(0.0)
        ).where(price_factors["f_res_mom_vol"].notna())

    if membership_mask is not None:
        aligned_mask = membership_mask.reindex(index=sig.index, columns=sig.columns).fillna(False).astype(bool)
        sig = sig.where(aligned_mask)

    close_w = price_factors["close_w"]
    next_open_w = price_factors.get("next_open_w")
    ret_1w = price_factors["ret_1w"]
    ew_1w = ret_1w.mean(axis=1)
    tgt_4w = price_factors["target_4w"]

    valid_dates = sig.dropna(how="all").index
    if len(valid_dates) < 20:
        return {"IC": np.nan, "IC_t_NW6": np.nan, "Port_CAGR": np.nan, "EW_CAGR": np.nan, "True_MDD": np.nan, "Sharpe": np.nan}

    ic_vals = []
    for dt in valid_dates:
        if dt not in tgt_4w.index:
            continue
        a, b = sig.loc[dt], tgt_4w.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() >= 6 and a[m].nunique() > 1:
            ic_vals.append(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))
    ic_s = pd.Series(ic_vals, dtype=float)

    cohort_rets = []
    for offset in range(4):
        held = set()
        wk = {}
        for i in range(len(valid_dates) - 1):
            dt, nxt = valid_dates[i], valid_dates[i + 1]
            cost = 0.0
            new_entries = set()
            exited = set()
            if i % 4 == offset or not held:
                a = sig.loc[dt].dropna().sort_values(ascending=False)
                k = max(2, int(np.floor(len(a) * buy_frac)))
                new_set = set(a.index[:k])
                if held:
                    new_entries = new_set - held
                    exited = held - new_set
                    turn = len(new_entries) / max(len(new_set), 1)
                    cost = turn * round_trip_cost
                else:
                    new_entries = set(new_set)
                held = new_set

            if held:
                per_stock_rets = []
                retained = held - new_entries
                if retained:
                    r_ret = ret_1w.loc[nxt, list(retained)].dropna()
                    per_stock_rets.extend(r_ret.tolist())

                if new_entries:
                    if next_open_w is not None and dt in next_open_w.index:
                        ent_open = next_open_w.loc[dt, list(new_entries)]
                        ent_close = close_w.loc[nxt, list(new_entries)]
                        r_ent = (ent_close / ent_open.replace(0.0, np.nan) - 1.0).dropna()
                        if exited:
                            ex_open = next_open_w.loc[dt, list(exited)]
                            ex_prev = close_w.loc[dt, list(exited)]
                            gap_exit = (ex_open / ex_prev.replace(0.0, np.nan) - 1.0).dropna()
                            gap_mean = float(gap_exit.mean()) if len(gap_exit) else 0.0
                            r_ent = (1.0 + gap_mean) * (1.0 + r_ent) - 1.0
                        per_stock_rets.extend(r_ent.tolist())
                    else:
                        r_ent = ret_1w.loc[nxt, list(new_entries)].dropna()
                        per_stock_rets.extend(r_ent.tolist())

                gross_r = float(np.mean(per_stock_rets)) if per_stock_rets else 0.0
                wk[nxt] = gross_r - cost
        cohort_rets.append(pd.Series(wk))

    port_w = pd.concat(cohort_rets, axis=1).mean(axis=1).dropna()
    ew_w = ew_1w.reindex(port_w.index).dropna()
    port_w = port_w.reindex(ew_w.index)
    eq = (1.0 + port_w).cumprod()
    ew_eq = (1.0 + ew_w).cumprod()
    n_w = len(port_w)
    cagr = float(eq.iloc[-1] ** (52.0 / n_w) - 1.0) if n_w > 0 else np.nan
    ew_cagr = float(ew_eq.iloc[-1] ** (52.0 / n_w) - 1.0) if n_w > 0 else np.nan
    mdd = float((eq / eq.cummax() - 1.0).min()) if n_w > 0 else np.nan
    sharpe = float(port_w.mean() / port_w.std() * np.sqrt(52.0)) if port_w.std() > 0 else np.nan

    return {
        "Model": model.lower(),
        "IC": float(ic_s.mean()) if len(ic_s) else np.nan,
        "IC_t_NW6": float(ev.newey_west_t(ic_s, lags=NW_LAGS)) if len(ic_s) >= 6 else np.nan,
        "Port_CAGR": cagr,
        "EW_CAGR": ew_cagr,
        "True_MDD": mdd,
        "Sharpe": sharpe,
        "Weeks": n_w,
        "Execution": "Next-Day Open",
    }


def make_offline_demo_data(groups: dict) -> tuple:
    """Fast vectorized deterministic synthetic multi-market dataset for offline testing and demos."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=780)
    n_dates = len(dates)
    mkt_ret = rng.normal(0.0004, 0.010, n_dates)
    bench = pd.Series(100.0 * np.cumprod(1.0 + mkt_ret), index=dates, name="BENCH")

    tickers = list(dict.fromkeys(t for mem in groups.values() for t in mem))
    n_t = len(tickers)
    idx_arr = np.arange(n_t)
    drifts = 0.0002 + (idx_arr % 5) * 0.00015
    noise = rng.normal(0.0, 0.014, size=(n_dates, n_t))
    rets = 0.9 * mkt_ret[:, None] + drifts[None, :] + noise
    starts = 50.0 * (1 + (idx_arr % 4))
    prices = starts[None, :] * np.cumprod(1.0 + rets, axis=0)
    close_df = pd.DataFrame(prices, index=dates, columns=tickers)
    overnight = rng.normal(0.0, 0.003, size=(n_dates, n_t))
    prev_close = np.vstack([starts[None, :], prices[:-1, :]])
    open_df = pd.DataFrame(prev_close * (1.0 + overnight), index=dates, columns=tickers)

    fund_df = pd.DataFrame({
        "ticker": tickers,
        "name": tickers,
        "sector": "",
        "currency": [infer_ticker_currency(t) for t in tickers],
        "dividend_yield": np.round(1.0 + (idx_arr % 7) * 1.1, 2),
        "PER": np.round(8.0 + (idx_arr % 6) * 4.5, 2),
        "PBR": np.round(0.8 + (idx_arr % 5) * 0.9, 2),
        "ev_ebitda": np.round(5.0 + (idx_arr % 7) * 2.5, 2),
        "rev_growth": np.round(-0.05 + (idx_arr % 6) * 0.08, 3),
        "earnings_growth": np.round(-0.08 + (idx_arr % 7) * 0.09, 3),
        "roe": np.round(0.05 + (idx_arr % 6) * 0.05, 3),
        "op_margin": np.round(0.06 + (idx_arr % 5) * 0.06, 3),
        "gross_margin": np.round(0.20 + (idx_arr % 6) * 0.10, 3),
        "fcf_yield": np.round(0.01 + (idx_arr % 6) * 0.02, 3),
        "debt_to_equity": np.round(20.0 + (idx_arr % 5) * 35.0, 1),
    }).set_index("ticker")
    return close_df, open_df, bench, fund_df


def resolve_universe_selection(
    market: str = "",
    industry: str = "",
    universe_file: Optional[Path] = None,
    tickers_csv: str = "",
    benchmark_override: str = "",
) -> Tuple[dict, list, str, float, str, str, bool]:
    """Resolve CLI flags into (groups, tickers, benchmark, cost, mkt_label, mkt_name, convert_to_usd)."""
    if tickers_csv.strip():
        custom_list = [t.strip() for t in tickers_csv.split(",") if t.strip()]
        groups = {"custom": tuple(custom_list)}
        tickers = custom_list
        benchmark = benchmark_override or "ACWI"
        cost = 0.0020
        mkt_label = "CUSTOM"
        mkt_name = "Custom Watchlist"
        ccys = {infer_ticker_currency(t) for t in tickers}
        convert_usd = len(ccys) > 1 or (benchmark.upper() in {"ACWI", "SPY", "VT"} and ccys != {"USD"})
        return groups, tickers, benchmark, cost, mkt_label, mkt_name, convert_usd

    if universe_file is not None and not market.strip():
        cfg, groups = load_universe_file(universe_file)
        if industry.strip():
            if industry.strip() not in groups:
                raise ValueError(f"Industry '{industry}' not found in {universe_file}.")
            groups = {industry.strip(): groups[industry.strip()]}
        tickers = list(dict.fromkeys(t for mems in groups.values() for t in mems))
        benchmark = benchmark_override or cfg["benchmark"]
        cost = cfg["round_trip_cost"]
        mkt_label = universe_file.stem.upper()
        mkt_name = cfg["name"]
        ccys = {infer_ticker_currency(t) for t in tickers}
        convert_usd = cfg.get("currency", "USD").upper() == "USD" and ccys != {"USD"}
        return groups, tickers, benchmark, cost, mkt_label, mkt_name, convert_usd

    mkt_key = (market.strip() or "GLOBAL").upper()
    if mkt_key not in MARKET_CONFIG:
        raise ValueError(f"Unsupported market '{mkt_key}'. Choose from: {', '.join(list_markets())}")
    cfg = MARKET_CONFIG[mkt_key]
    groups = get_market_groups(mkt_key, industry=industry or None, extra_file=universe_file)
    tickers = list(dict.fromkeys(t for mems in groups.values() for t in mems))
    benchmark = benchmark_override or cfg["benchmark"]
    cost = cfg["round_trip_cost"]
    mkt_label = f"{mkt_key}/{industry}" if industry else mkt_key
    mkt_name = cfg["name"]
    convert_usd = (mkt_key == "GLOBAL") or (cfg.get("currency", "USD").upper() == "USD" and any(infer_ticker_currency(t) != "USD" for t in tickers))
    return groups, tickers, benchmark, cost, mkt_label, mkt_name, convert_usd


def load_membership_csv(path: Path) -> pd.DataFrame:
    """Load optional point-in-time boolean constituent mask CSV (date index, ticker columns)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball Global Equity Multi-Factor Ranker & Backtester (v6/v5)")
    parser.add_argument("--market", default="", help=f"Market preset ({', '.join(list_markets())}; defaults to GLOBAL if omitted without --universe-file)")
    parser.add_argument("--model", default="v6", choices=["v6", "v5"], help="Empirical scoring model (default: v6 6-pillar)")
    parser.add_argument("--industry", default="", help="Optional single industry filter (e.g. semiconductors_hardware)")
    parser.add_argument("--universe-file", type=Path, default=None, help="Standalone custom JSON universe file (or merged into --market if both provided)")
    parser.add_argument("--membership-csv", type=Path, default=None, help="Optional point-in-time historical constituent mask CSV to eliminate survivorship bias")
    parser.add_argument("--tickers", default="", help="Optional comma-separated custom tickers (e.g. AAPL,MSFT,7203.T,2330.TW)")
    parser.add_argument("--benchmark", default="", help="Override benchmark ticker (e.g. SPY, ACWI, ^N225, ^STOXX50E)")
    parser.add_argument("--period", default="5y", help="Yahoo Finance history window (default: 5y)")
    parser.add_argument("--top", type=int, default=20, help="Number of top-ranked stocks to display")
    parser.add_argument("--skip-fundamentals", action="store_true", help="Skip per-ticker Yahoo fundamental fetch (faster for 250+ tickers)")
    parser.add_argument("--backtest", action="store_true", help="Also run 4-cohort staggered historical backtest (Next-Day Open execution)")
    parser.add_argument("--offline-demo", action="store_true", help="Run on synthetic data without network")
    args = parser.parse_args()

    groups, tickers, benchmark, cost, mkt_label, mkt_name, convert_usd = resolve_universe_selection(
        market=args.market,
        industry=args.industry,
        universe_file=args.universe_file,
        tickers_csv=args.tickers,
        benchmark_override=args.benchmark,
    )

    if args.offline_demo:
        close_d, open_d, bench_d, fund_df = make_offline_demo_data(groups)
        mode_tag = f"OFFLINE DEMO ({len(tickers)} stocks)"
    else:
        close_d, open_d, bench_d = download_global_prices(tickers, benchmark, period=args.period)
        if convert_usd:
            close_d, open_d = convert_prices_to_usd(close_d, open_d, period=args.period)
        fund_df = (
            make_neutral_fundamentals(list(close_d.columns))
            if args.skip_fundamentals
            else fetch_global_fundamentals(list(close_d.columns))
        )
        fx_note = ", USD-normalized" if convert_usd else ""
        mode_tag = f"LIVE ({mkt_name}, {len(close_d.columns)}/{len(tickers)} stocks{fx_note})"

    pf = compute_global_price_factors(close_d, bench_d, groups, open_d=open_d)
    ranked = score_latest_global(pf, fund_df, groups, model=args.model)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 18)
    print(f"=== Snowball Global {args.model.upper()} Ranking | {mkt_label} | {mode_tag} | Benchmark: {benchmark} ===")
    print(ranked.head(args.top).round(3).to_string(index=False))

    if args.backtest:
        mem_mask = load_membership_csv(args.membership_csv) if args.membership_csv else None
        if mem_mask is None:
            print(
                "\n[NOTICE] Survivorship Bias Disclosure: Static current-day universe omits delisted stocks; "
                "historical CAGR/Sharpe are upward-biased. Pass --membership-csv for point-in-time constituents."
            )
        bt_res = backtest_global_staggered(pf, round_trip_cost=cost, membership_mask=mem_mask, model=args.model)
        print(f"\n=== 4-Cohort Staggered Pure-Price {args.model.upper()} Backtest (Next-Day Open Fill) ===")
        for k, v in bt_res.items():
            if isinstance(v, float):
                print(f"  {k:<12}: {v:.4f}")
            else:
                print(f"  {k:<12}: {v}")


if __name__ == "__main__":
    main()
