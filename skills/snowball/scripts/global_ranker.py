#!/usr/bin/env python3
"""Universal Global Equity v5 Multi-Factor Ranker & Point-in-Time Backtester.

Works across ANY Yahoo Finance equity market (US, EU, JP, HK, TW, GLOBAL — 1,340+ built-in stocks
across 13 industries with >= 20 stocks/industry and >= 250 stocks/market), custom JSON universe
files, or arbitrary comma-separated tickers:
1. Residual Momentum (F_ResMom_26_4w, Blitz et al. 2011): 26w skipping 4w Beta-adjusted alpha.
2. Full-Group Industry Momentum (F_IndMom_Full_12w, Moskowitz & Grinblatt 1999).
3. Fundamental Growth (F_Fund): Revenue growth snapshot (live ranking) or neutral in
   pure-price historical mode.
4. 8%-Capped Industry-Neutral Value (F_NormVal): Within-sector 1/PBR (financials) or capped 1/PER
   (non-financials, E/P <= 15%) + 0.5 * capped dividend yield (Yield <= 8.0%).

Usage:
    python global_ranker.py --market US --top 20
    python global_ranker.py --market US --industry semiconductors_hardware
    python global_ranker.py --market GLOBAL --backtest --skip-fundamentals
    python global_ranker.py --universe-file ./my_extra_universe.json
    python global_ranker.py --tickers AAPL,MSFT,GOOGL,ASML.AS,7203.T,2330.TW --benchmark ACWI
    python global_ranker.py --market US --offline-demo
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


def download_global_prices(
    tickers: list,
    benchmark: str,
    period: str = "5y",
    batch_size: int = 80,
) -> tuple:
    """Download dividend-adjusted daily close prices in batches from Yahoo Finance."""
    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        raise ValueError("No tickers provided.")

    frames: List[pd.DataFrame] = []
    # Download benchmark cleanly first
    bench_raw = yf.download([benchmark], period=period, progress=False, auto_adjust=True, threads=False)
    if bench_raw is None or bench_raw.empty or "Close" not in bench_raw:
        raise RuntimeError(f"Benchmark '{benchmark}' returned no price history.")
    bench_close = bench_raw["Close"]
    if isinstance(bench_close, pd.DataFrame):
        bench_close = bench_close.iloc[:, 0]
    bench_series = bench_close.dropna().rename(benchmark)
    if bench_series.empty:
        raise RuntimeError(f"Benchmark '{benchmark}' returned no price history.")

    for start in range(0, len(unique_tickers), batch_size):
        chunk = unique_tickers[start:start + batch_size]
        raw = yf.download(chunk, period=period, progress=False, auto_adjust=True, threads=True)
        if raw is None or raw.empty or "Close" not in raw:
            continue
        closes = raw["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=chunk[0])
        frames.append(closes)

    if not frames:
        raise RuntimeError(f"Unable to fetch Yahoo Finance prices for {len(unique_tickers)} symbols.")

    all_closes = pd.concat(frames, axis=1)
    all_closes = all_closes.loc[:, ~all_closes.columns.duplicated()]
    stock_cols = [t for t in unique_tickers if t in all_closes.columns and all_closes[t].dropna().shape[0] >= 30]
    if not stock_cols:
        raise RuntimeError("No stock symbols returned sufficient price history.")
    return all_closes[stock_cols], bench_series


def _fetch_one_fundamental(sym: str) -> dict:
    try:
        info = yf.Ticker(sym).info or {}
    except Exception:
        info = {}
    dy = info.get("dividendYield")
    if dy is not None:
        dy_pct = float(dy) * 100.0 if float(dy) < 0.5 else float(dy)
    else:
        dy_pct = 0.0
    per = info.get("trailingPE") or info.get("forwardPE")
    pbr = info.get("priceToBook")
    rev_g = info.get("revenueGrowth")
    return {
        "ticker": sym,
        "name": info.get("shortName") or sym,
        "sector": info.get("sector") or "",
        "currency": info.get("currency") or "",
        "dividend_yield": max(0.0, dy_pct),
        "PER": float(per) if per is not None and float(per) > 0 else np.nan,
        "PBR": float(pbr) if pbr is not None and float(pbr) > 0 else np.nan,
        "rev_growth": float(rev_g) if rev_g is not None else np.nan,
    }


def fetch_global_fundamentals(tickers: list, max_workers: int = 8) -> pd.DataFrame:
    """Fetch point-in-time snapshot fundamentals concurrently from Yahoo Finance."""
    if not tickers:
        return pd.DataFrame(columns=["name", "sector", "currency", "dividend_yield", "PER", "PBR", "rev_growth"])
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
        "currency": "",
        "dividend_yield": 0.0,
        "PER": np.nan,
        "PBR": np.nan,
        "rev_growth": np.nan,
    }).set_index("ticker")


def compute_global_price_factors(close_d: pd.DataFrame, bench_d: pd.Series, groups: dict) -> dict:
    """Compute weekly point-in-time Residual Momentum and Full-Group Industry Momentum."""
    close_w = close_d.resample("W-FRI").last().ffill()
    bench_w = bench_d.resample("W-FRI").last().ffill().reindex(close_w.index)

    ret_1w = close_w.pct_change(fill_method=None)
    mkt_1w = bench_w.pct_change(fill_method=None)

    beta = ft.rolling_beta(ret_1w, mkt_1w, BETA_WINDOW, BETA_MIN)
    alpha = ret_1w - beta.shift(1).mul(mkt_1w, axis=0)
    target_4w = sum(alpha.shift(-k) for k in range(1, 5))

    # 1. Residual Momentum (Blitz et al. 2011): 26w skipping last 4w cumulative alpha
    res_mom_raw = alpha.shift(4).rolling(22, min_periods=12).sum()
    f_res_mom = ft.cs_zscore(res_mom_raw)

    # 2. Full-Group Industry Momentum (12w inclusive group mean alpha)
    alpha_12w = alpha.rolling(12, min_periods=6).sum()
    f_ind_mom = ft.cs_zscore(_group_mean_incl(alpha_12w, groups))

    # 3. Low Idiosyncratic Volatility (defensive tie-breaker for pure-price backtest)
    ivol = alpha.rolling(26, min_periods=13).std()
    f_low_vol = ft.cs_zscore(-ivol)

    return {
        "close_w": close_w,
        "ret_1w": ret_1w,
        "mkt_1w": mkt_1w,
        "alpha": alpha,
        "target_4w": target_4w,
        "f_res_mom": f_res_mom,
        "f_ind_mom": f_ind_mom,
        "f_low_vol": f_low_vol,
    }


def score_latest_global(
    price_factors: dict,
    fundamentals: pd.DataFrame,
    groups: dict,
) -> pd.DataFrame:
    """Combine latest weekly Residual Momentum, Industry Momentum, Revenue Growth, and 8%-Capped Value."""
    dt = price_factors["f_res_mom"].index[-1]
    tickers = list(price_factors["close_w"].columns)
    gmap = group_map_for(groups)

    f_res = price_factors["f_res_mom"].loc[dt].reindex(tickers).fillna(0.0)
    f_ind = price_factors["f_ind_mom"].loc[dt].reindex(tickers).fillna(0.0)

    fund = fundamentals.reindex(tickers)
    ey = (1.0 / fund["PER"]).clip(upper=EY_CAP)
    bp = 1.0 / fund["PBR"]
    yld_cap = fund["dividend_yield"].fillna(0.0).clip(upper=YIELD_CAP_PCT)

    core_val = pd.Series(
        {t: (bp[t] if gmap.get(t) in GLOBAL_PB_GROUPS else ey[t]) for t in tickers},
        dtype=float,
    )
    # Within-group value z-score (fallback to cross-sectional if single-stock custom group)
    val_df = core_val.to_frame().T
    val_ind = _group_zscore(val_df, groups).iloc[0]
    if val_ind.isna().all():
        sd = core_val.std()
        val_ind = ((core_val - core_val.mean()) / sd).clip(-3.0, 3.0) if (sd and sd > 0) else pd.Series(0.0, index=tickers)
    val_ind = val_ind.fillna(0.0)

    yld_sd = yld_cap.std()
    val_yld = ((yld_cap - yld_cap.mean()) / yld_sd).clip(-3.0, 3.0) if (yld_sd and yld_sd > 0) else pd.Series(0.0, index=tickers)
    raw_val = val_ind + 0.5 * val_yld
    rv_sd = raw_val.std()
    f_norm_val = ((raw_val - raw_val.mean()) / rv_sd).clip(-3.0, 3.0) if (rv_sd and rv_sd > 0) else pd.Series(0.0, index=tickers)

    rev_g = fund["rev_growth"].clip(-0.5, 1.5)
    rev_sd = rev_g.std()
    f_fin = ((rev_g - rev_g.mean()) / rev_sd).clip(-3.0, 3.0).fillna(0.0) if (rev_sd and rev_sd > 0) else pd.Series(0.0, index=tickers)

    raw_score = 0.5 * f_res + 0.5 * f_ind + 1.0 * f_fin + 1.5 * f_norm_val
    rs_sd = raw_score.std()
    v5_score = ((raw_score - raw_score.mean()) / rs_sd).clip(-3.0, 3.0) if (rs_sd and rs_sd > 0) else pd.Series(0.0, index=tickers)

    out = pd.DataFrame({
        "ticker": tickers,
        "name": fund["name"].fillna(pd.Series(tickers, index=tickers)).values,
        "group": [gmap.get(t, "custom") for t in tickers],
        "close": price_factors["close_w"].loc[dt, tickers].values,
        "yield_%": fund["dividend_yield"].fillna(0.0).values,
        "PER": fund["PER"].values,
        "rev_yoy_%": (fund["rev_growth"] * 100.0).values,
        "res_mom": f_res.values,
        "ind_mom": f_ind.values,
        "norm_val": f_norm_val.values,
        "v5_score": v5_score.values,
    }).sort_values("v5_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def backtest_global_staggered(
    price_factors: dict,
    round_trip_cost: float = 0.0020,
    buy_frac: float = 0.20,
) -> dict:
    """4-cohort staggered historical backtest for global pure-price v5 signal (ResMom + IndMom)."""
    sig = ft.cs_zscore(
        price_factors["f_res_mom"].fillna(0.0) + price_factors["f_ind_mom"].fillna(0.0)
    ).where(price_factors["f_res_mom"].notna())
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
            if i % 4 == offset or not held:
                a = sig.loc[dt].dropna().sort_values(ascending=False)
                k = max(2, int(np.floor(len(a) * buy_frac)))
                new_set = set(a.index[:k])
                if held:
                    turn = len(new_set - held) / max(len(new_set), 1)
                    cost = turn * round_trip_cost
                held = new_set
            if held:
                r = ret_1w.loc[nxt, list(held)].dropna()
                wk[nxt] = (float(r.mean()) if len(r) else 0.0) - cost
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
        "IC": float(ic_s.mean()) if len(ic_s) else np.nan,
        "IC_t_NW6": float(ev.newey_west_t(ic_s, lags=NW_LAGS)) if len(ic_s) >= 6 else np.nan,
        "Port_CAGR": cagr,
        "EW_CAGR": ew_cagr,
        "True_MDD": mdd,
        "Sharpe": sharpe,
        "Weeks": n_w,
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

    fund_df = pd.DataFrame({
        "ticker": tickers,
        "name": tickers,
        "sector": "",
        "currency": "USD",
        "dividend_yield": np.round(1.0 + (idx_arr % 7) * 1.1, 2),
        "PER": np.round(8.0 + (idx_arr % 6) * 4.5, 2),
        "PBR": np.round(0.8 + (idx_arr % 5) * 0.9, 2),
        "rev_growth": np.round(-0.05 + (idx_arr % 6) * 0.08, 3),
    }).set_index("ticker")
    return close_df, bench, fund_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball Global Equity v5 Multi-Factor Ranker & Backtester")
    parser.add_argument("--market", default="GLOBAL", help=f"Market preset ({', '.join(list_markets())})")
    parser.add_argument("--industry", default="", help="Optional single industry filter (e.g. semiconductors_hardware)")
    parser.add_argument("--universe-file", type=Path, default=None, help="Optional custom JSON universe file to merge or rank")
    parser.add_argument("--tickers", default="", help="Optional comma-separated custom tickers (e.g. AAPL,MSFT,7203.T,2330.TW)")
    parser.add_argument("--benchmark", default="", help="Override benchmark ticker (e.g. SPY, ACWI, ^N225, ^STOXX50E)")
    parser.add_argument("--period", default="5y", help="Yahoo Finance history window (default: 5y)")
    parser.add_argument("--top", type=int, default=20, help="Number of top-ranked stocks to display")
    parser.add_argument("--skip-fundamentals", action="store_true", help="Skip per-ticker Yahoo fundamental fetch (faster for 250+ tickers)")
    parser.add_argument("--backtest", action="store_true", help="Also run 4-cohort staggered historical backtest")
    parser.add_argument("--offline-demo", action="store_true", help="Run on synthetic data without network")
    args = parser.parse_args()

    if args.tickers.strip():
        custom_list = [t.strip() for t in args.tickers.split(",") if t.strip()]
        groups = {"custom": tuple(custom_list)}
        tickers = custom_list
        benchmark = args.benchmark or "ACWI"
        cost = 0.0020
        mkt_label = "CUSTOM"
        mkt_name = "Custom Watchlist"
    elif args.universe_file is not None and not args.market:
        cfg, groups = load_universe_file(args.universe_file)
        tickers = list(dict.fromkeys(t for mems in groups.values() for t in mems))
        benchmark = args.benchmark or cfg["benchmark"]
        cost = cfg["round_trip_cost"]
        mkt_label = args.universe_file.stem.upper()
        mkt_name = cfg["name"]
    else:
        mkt_key = args.market.upper()
        if mkt_key not in MARKET_CONFIG:
            raise ValueError(f"Unsupported market '{args.market}'. Choose from: {', '.join(list_markets())}")
        cfg = MARKET_CONFIG[mkt_key]
        groups = get_market_groups(mkt_key, industry=args.industry or None, extra_file=args.universe_file)
        tickers = list(dict.fromkeys(t for mems in groups.values() for t in mems))
        benchmark = args.benchmark or cfg["benchmark"]
        cost = cfg["round_trip_cost"]
        mkt_label = f"{mkt_key}/{args.industry}" if args.industry else mkt_key
        mkt_name = cfg["name"]

    if args.offline_demo:
        close_d, bench_d, fund_df = make_offline_demo_data(groups)
        mode_tag = f"OFFLINE DEMO ({len(tickers)} stocks)"
    else:
        close_d, bench_d = download_global_prices(tickers, benchmark, period=args.period)
        fund_df = (
            make_neutral_fundamentals(list(close_d.columns))
            if args.skip_fundamentals
            else fetch_global_fundamentals(list(close_d.columns))
        )
        mode_tag = f"LIVE ({mkt_name}, {len(close_d.columns)}/{len(tickers)} stocks)"

    pf = compute_global_price_factors(close_d, bench_d, groups)
    ranked = score_latest_global(pf, fund_df, groups)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 15)
    print(f"=== Snowball Global v5 Ranking | {mkt_label} | {mode_tag} | Benchmark: {benchmark} ===")
    print(ranked.head(args.top).round(3).to_string(index=False))

    if args.backtest:
        bt_res = backtest_global_staggered(pf, round_trip_cost=cost)
        print("\n=== 4-Cohort Staggered Pure-Price v5 Backtest (ResMom + IndMom) ===")
        for k, v in bt_res.items():
            if isinstance(v, float):
                print(f"  {k:<12}: {v:.4f}")
            else:
                print(f"  {k:<12}: {v}")


if __name__ == "__main__":
    main()
