"""Point-in-Time 2,000-Stock (US + TW) Multi-Market Training Panel Builder for Kaggle & Local ML.

Builds a strictly leak-free weekly panel combining US (1,000) + TW (1,000) equities (or arbitrary
market combinations from Snowball's official universes) for Purged Walk-Forward training:

- Zero Look-Ahead Price & Macro Features:
  * Volatility-scaled 12-1m Residual Momentum (`f_res_mom_vol`) + Frog-in-the-Pan (`f_fip`)
  * Full-group Industry Momentum (`f_ind_mom`, within market × industry)
  * Low-Risk / BAB + IVOL (`f_low_risk`)
  * Short-term 1-week reversal (`f_rev_1w`), 3-month momentum (`f_mom_3m`), realized volatility (`f_vol_60d`)
  * Dollar-volume liquidity (`f_liq`) when volume is available
  * Strictly lagged `shift(1)` macro regime features (`macro_stress`, `rate_inflation`, `term_spread`, `vix`)
  * Macro × Micro interaction terms (`ix_stress_x_lowrisk`, `ix_stress_x_mom`)
- Optional Point-in-Time Historical Fundamentals (`--pit-fundamentals-csv`):
  * Merged strictly via backward `merge_asof` on `available_date` (filing date + lag), never using
    static current-day Yahoo `.info` snapshots for historical rows!
- Realistic Next-Trading-Day Open Forward Labels (H=4w and H=12w):
  * Entry = next trading day's open strictly after decision Friday t
  * Exit  = next trading day's open strictly after decision Friday t + H
  * `fwd_ret_4w`, `fwd_excess_4w` (vs regional benchmark SPY / 0050.TW), `fwd_vol_adj_4w`
  * `target_rank_4w`: cross-sectional percentile rank in [0, 1] per (date, market)

Usage:
    # Build live 2,000-stock US+TW panel (Parquet on Kaggle / CSV.gz fallback locally)
    python build_training_panel.py --markets US,TW --period 8y --output ./kaggle_dataset/snowball_us_tw_2000.parquet

    # Fast deterministic offline demo (2,000 stocks synthetic)
    python build_training_panel.py --markets US,TW --offline-demo --output ./kaggle_dataset/snowball_us_tw_2000.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import features as ft
from global_ranker import (
    compute_global_price_factors,
    compute_next_open_weekly,
    convert_prices_to_usd,
)
from global_universe import MARKET_CONFIG, get_market_groups
from macro_micro import (
    compute_macro_regime_panel,
    download_macro_series,
    make_offline_macro_data,
)

# Regional ETF / Index benchmarks for clean USD-comparable excess return labels
REGIONAL_BENCHMARKS: Dict[str, str] = {
    "US": "SPY",
    "TW": "0050.TW",
    "EU": "^STOXX50E",
    "JP": "^N225",
    "HK": "^HSI",
}

CORE_FEATURE_COLS: Tuple[str, ...] = (
    "f_mom_qual",
    "f_res_mom_vol",
    "f_fip",
    "f_ind_mom",
    "f_low_risk",
    "f_rev_1w",
    "f_mom_3m",
    "f_vol_60d",
    "f_liq",
    "macro_stress",
    "rate_inflation",
    "ix_stress_x_lowrisk",
    "ix_stress_x_mom",
    "is_tw",
)

PIT_FUNDAMENTAL_COLS: Tuple[str, ...] = (
    "pit_rule40_z",
    "pit_rev_yoy_z",
    "pit_gross_margin_chg_z",
    "pit_roe_z",
    "pit_ep_z",
    "pit_fcf_margin_z",
)


def _has_parquet_engine() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Save DataFrame to Parquet when available, gracefully falling back to compressed CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if _has_parquet_engine():
            df.to_parquet(path, index=False, compression="snappy")
            return path
        fallback = path.with_suffix(".csv.gz")
        df.to_csv(fallback, index=False, compression="gzip")
        print(f"[INFO] pyarrow/fastparquet not installed locally; saved compressed CSV instead: {fallback}")
        return fallback
    if path.name.endswith(".csv.gz"):
        df.to_csv(path, index=False, compression="gzip")
        return path
    if suffix == ".pkl":
        df.to_pickle(path)
        return path
    df.to_csv(path, index=False)
    return path


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load Parquet, CSV.gz, CSV, or Pickle transparently (with auto fallback if .parquet -> .csv.gz)."""
    target = path
    if not target.exists() and target.suffix.lower() == ".parquet":
        alt = target.with_suffix(".csv.gz")
        if alt.exists():
            target = alt
    if not target.exists():
        raise FileNotFoundError(f"Training panel file not found: {path}")

    if target.suffix.lower() == ".parquet":
        df = pd.read_parquet(target)
    elif target.suffix.lower() == ".pkl":
        df = pd.read_pickle(target)
    else:
        df = pd.read_csv(target)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _download_batched_ohlcv(
    tickers: Sequence[str],
    period: str = "8y",
    batch_size: int = 150,
    pause_sec: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download adjusted Close, unadjusted nominal Raw Close, adjusted Open, and Volume in batches.

    Using `auto_adjust=False` preserves unadjusted nominal `raw_close` for split-immune Point-in-Time
    US `ttm_eps / raw_close` valuation ratios, while keeping total-return `Adj Close` and
    `Open * (Adj Close / Close)` for momentum and Next-Day Open return labels.
    """
    uniq = list(dict.fromkeys(tickers))
    close_parts: List[pd.DataFrame] = []
    raw_close_parts: List[pd.DataFrame] = []
    open_parts: List[pd.DataFrame] = []
    vol_parts: List[pd.DataFrame] = []

    n_batches = (len(uniq) + batch_size - 1) // batch_size
    for idx in range(0, len(uniq), batch_size):
        chunk = uniq[idx : idx + batch_size]
        b_num = idx // batch_size + 1
        print(f"  [batch {b_num}/{n_batches}] downloading {len(chunk)} tickers ({period})...")
        try:
            raw = yf.download(chunk, period=period, progress=False, auto_adjust=False, threads=True)
            if raw is None or raw.empty or "Close" not in raw:
                continue
            rc = raw["Close"]
            ac = raw["Adj Close"] if "Adj Close" in raw else rc.copy()
            ro = raw["Open"] if "Open" in raw else rc.copy()
            v = raw["Volume"] if "Volume" in raw else pd.DataFrame(np.nan, index=rc.index, columns=rc.columns)
            if isinstance(rc, pd.Series):
                rc = rc.to_frame(name=chunk[0])
                ac = ac.to_frame(name=chunk[0])
                ro = ro.to_frame(name=chunk[0])
                v = v.to_frame(name=chunk[0])
            adj_factor = (ac / rc.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            ao = ro * adj_factor
            close_parts.append(ac)
            raw_close_parts.append(rc)
            open_parts.append(ao)
            vol_parts.append(v)
        except Exception as exc:
            print(f"  [WARN] batch {b_num} failed ({exc}); skipping chunk.")
        if pause_sec > 0 and idx + batch_size < len(uniq):
            time.sleep(pause_sec)

    if not close_parts:
        raise RuntimeError("No historical price data downloaded from Yahoo Finance.")

    close_d = pd.concat(close_parts, axis=1)
    raw_close_d = pd.concat(raw_close_parts, axis=1)
    open_d = pd.concat(open_parts, axis=1)
    vol_d = pd.concat(vol_parts, axis=1)
    # Remove any duplicate columns safely
    close_d = close_d.loc[:, ~close_d.columns.duplicated()].dropna(how="all")
    raw_close_d = raw_close_d.loc[:, ~raw_close_d.columns.duplicated()].reindex(index=close_d.index, columns=close_d.columns)
    open_d = open_d.loc[:, ~open_d.columns.duplicated()].reindex(index=close_d.index, columns=close_d.columns)
    vol_d = vol_d.loc[:, ~vol_d.columns.duplicated()].reindex(index=close_d.index, columns=close_d.columns)
    return close_d, raw_close_d, open_d, vol_d


def _make_synthetic_market_data(
    market: str,
    groups: Dict[str, Tuple[str, ...]],
    n_days: int = 1040,
    seed: int = 42,
    max_stocks: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, Tuple[str, ...]]]:
    """Generate realistic synthetic daily OHLCV with embedded weak persistent factor alpha + noise.

    Ensures offline tests and demos exercise realistic signal-to-noise (Rank IC ~ 0.03–0.08)
    without any data leakage.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Optionally trim per-industry for ultra-fast unit tests while preserving all 13 industries
    trimmed_groups: Dict[str, Tuple[str, ...]] = {}
    all_tickers: List[str] = []
    if max_stocks is not None and max_stocks > 0:
        per_ind = max(3, int(np.ceil(max_stocks / max(len(groups), 1))))
        for ind, mems in groups.items():
            sub = tuple(mems[:per_ind])
            trimmed_groups[ind] = sub
            all_tickers.extend(sub)
        all_tickers = list(dict.fromkeys(all_tickers))[:max_stocks]
        allowed = set(all_tickers)
        trimmed_groups = {k: tuple(t for t in v if t in allowed) for k, v in trimmed_groups.items()}
        trimmed_groups = {k: v for k, v in trimmed_groups.items() if v}
    else:
        trimmed_groups = dict(groups)
        all_tickers = list(dict.fromkeys(t for mems in groups.values() for t in mems))

    n_t = len(all_tickers)
    mkt_ret = rng.normal(0.00035, 0.011, n_days)
    bench = pd.Series(100.0 * np.cumprod(1.0 + mkt_ret), index=dates, name=REGIONAL_BENCHMARKS.get(market, "SPY"))

    # Assign industry-level slow-moving cycles + stock-level persistent quality/momentum factor
    ind_names = list(trimmed_groups.keys())
    ind_idx_map = {t: i for i, ind in enumerate(ind_names) for t in trimmed_groups[ind]}
    ind_ids = np.array([ind_idx_map.get(t, 0) for t in all_tickers])

    # Industry daily factor (persistent autoregressive cycles so industry momentum has true weak signal)
    n_ind = max(len(ind_names), 1)
    ind_regime = np.zeros((n_days, n_ind))
    for k in range(n_ind):
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(2.0, 5.0)
        t_axis = np.linspace(0.0, freq * np.pi, n_days)
        ind_regime[:, k] = 0.00035 * np.sin(t_axis + phase) + rng.normal(0.0, 0.004, n_days)

    # Stock-level slowly rotating alpha + beta dispersion
    betas = rng.uniform(0.65, 1.45, n_t)
    idio_vols = rng.uniform(0.010, 0.026, n_t)
    # Slow stock drift (rotates every ~180 days so momentum & low-vol both carry realistic signal)
    stock_phase = rng.uniform(0, 2 * np.pi, n_t)
    t_grid = np.linspace(0.0, 3.0 * np.pi, n_days)[:, None]
    stock_drift = 0.00045 * np.cos(t_grid + stock_phase[None, :]) - 0.00025 * (betas - 1.0)[None, :]

    noise = rng.normal(0.0, 1.0, (n_days, n_t)) * idio_vols[None, :]
    rets = (
        betas[None, :] * mkt_ret[:, None]
        + ind_regime[:, ind_ids]
        + stock_drift
        + noise
    )
    rets = np.clip(rets, -0.18, 0.18)

    base_px = rng.uniform(15.0, 220.0, n_t)
    close_arr = base_px[None, :] * np.cumprod(1.0 + rets, axis=0)
    # Realistic overnight open gap (next-day open differs from close)
    open_gap = rng.normal(0.0, 0.003, (n_days, n_t))
    prev_close = np.vstack([base_px[None, :], close_arr[:-1]])
    open_arr = prev_close * (1.0 + open_gap)
    vol_arr = rng.lognormal(mean=13.0, sigma=1.1, size=(n_days, n_t))

    close_d = pd.DataFrame(close_arr, index=dates, columns=all_tickers)
    open_d = pd.DataFrame(open_arr, index=dates, columns=all_tickers)
    vol_d = pd.DataFrame(vol_arr, index=dates, columns=all_tickers)
    return close_d, open_d, vol_d, bench, trimmed_groups


def _merge_pit_fundamentals(panel: pd.DataFrame, pit_csv: Path, max_staleness_days: int = 180) -> pd.DataFrame:
    """Strictly Point-in-Time backward merge of historical fundamentals by `available_date`.

    Enforces `tolerance=pd.Timedelta(days=max_staleness_days)` so stale filings expire cleanly.
    """
    pit = pd.read_csv(pit_csv)
    if "ticker" not in pit.columns or "available_date" not in pit.columns:
        raise ValueError("PIT fundamentals CSV must contain 'ticker' and 'available_date' columns.")
    pit["available_date"] = pd.to_datetime(pit["available_date"]).astype("datetime64[ns]")
    pit = pit.sort_values("available_date")

    raw_map = {
        "rule40_pct": "pit_rule40_z",
        "rev_yoy": "pit_rev_yoy_z",
        "gross_margin_chg": "pit_gross_margin_chg_z",
        "roe": "pit_roe_z",
        "ep_yield": "pit_ep_z",
        "fcf_margin": "pit_fcf_margin_z",
    }
    present_cols = [c for c in raw_map if c in pit.columns]
    if not present_cols:
        return panel

    left = panel.sort_values("date").reset_index(drop=True)
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        left,
        pit[["ticker", "available_date", *present_cols]],
        left_on="date",
        right_on="available_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta(days=max_staleness_days),
    )
    # Assert zero look-ahead: available_date <= date everywhere it's non-null
    valid_ad = merged["available_date"].notna()
    if valid_ad.any() and (merged.loc[valid_ad, "available_date"] > merged.loc[valid_ad, "date"]).any():
        raise AssertionError("Look-ahead violation detected in PIT fundamentals merge!")

    for raw_c in present_cols:
        z_col = raw_map[raw_c]
        # Cross-sectional within-(date, market, industry) z-score, clipped [-3, +3]
        grp = merged.groupby(["date", "market", "industry"])[raw_c]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0.0, np.nan)
        merged[z_col] = ((merged[raw_c] - mu) / sd).clip(-3.0, 3.0)

    drop_cols = ["available_date", *present_cols]
    return merged.drop(columns=[c for c in drop_cols if c in merged.columns])


def _build_single_market_panel(
    market: str,
    close_d: pd.DataFrame,
    open_d: pd.DataFrame,
    vol_d: Optional[pd.DataFrame],
    bench_d: pd.Series,
    groups: Dict[str, Tuple[str, ...]],
    macro_d: pd.DataFrame,
    horizons: Sequence[int] = (4, 12),
    raw_close_d: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute strictly point-in-time weekly features and Next-Day Open labels for one market."""
    pf = compute_global_price_factors(close_d, bench_d, groups, open_d=open_d)
    close_w = pf["close_w"]
    raw_close_w = (
        raw_close_d.resample("W-FRI").last().reindex(index=close_w.index, columns=close_w.columns)
        if raw_close_d is not None
        else close_w
    )
    next_open_w = pf["next_open_w"]
    weeks = close_w.index
    tickers = list(close_w.columns)

    # Extra pure-price features (strictly computed on <= t prices)
    ret_1w = pf["ret_1w"]
    f_rev_1w = ft.cs_zscore(-ret_1w)
    f_mom_3m = ft.cs_zscore(close_w / close_w.shift(13) - 1.0)
    vol_60d = close_d.pct_change().rolling(60, min_periods=30).std().resample("W-FRI").last().reindex(weeks) * np.sqrt(252.0)
    f_vol_60d = ft.cs_zscore(vol_60d)

    if vol_d is not None and not vol_d.empty:
        dollar_vol = (close_d * vol_d).rolling(20, min_periods=10).mean().resample("W-FRI").last().reindex(weeks)
        f_liq = ft.cs_zscore(np.log1p(dollar_vol.clip(lower=0.0)))
    else:
        f_liq = pd.DataFrame(0.0, index=weeks, columns=tickers)

    # Benchmark Next-Trading-Day Open weekly series (for exact regional excess returns)
    bench_df = bench_d.to_frame(name="BENCH")
    bench_next_open = compute_next_open_weekly(bench_df, weeks)["BENCH"]

    # Lagged macro regime panel (shift(1) already enforced inside compute_macro_regime_panel)
    macro_panel = compute_macro_regime_panel(macro_d, weeks)

    # Forward returns strictly from Next-Day Open(t) -> Next-Day Open(t + H)
    fwd_cols: Dict[str, pd.DataFrame] = {}
    vol_floor = vol_60d.clip(lower=0.10, upper=1.50).fillna(0.25)
    for h in horizons:
        stk_fwd = next_open_w.shift(-h) / next_open_w.replace(0.0, np.nan) - 1.0
        b_fwd = bench_next_open.shift(-h) / bench_next_open.replace(0.0, np.nan) - 1.0
        exc_fwd = stk_fwd.sub(b_fwd, axis=0)
        vadj_fwd = exc_fwd / vol_floor
        # Cross-sectional percentile rank within (date, market) in [0, 1]
        rank_fwd = vadj_fwd.rank(axis=1, pct=True)
        fwd_cols[f"fwd_ret_{h}w"] = stk_fwd
        fwd_cols[f"fwd_excess_{h}w"] = exc_fwd
        fwd_cols[f"fwd_vol_adj_{h}w"] = vadj_fwd
        fwd_cols[f"target_rank_{h}w"] = rank_fwd

    # 1-week forward return from Next-Day Open(t) -> Next-Day Open(t+1) for realistic TWR portfolio backtest
    fwd_1w_open = next_open_w.shift(-1) / next_open_w.replace(0.0, np.nan) - 1.0
    bench_1w_open = bench_next_open.shift(-1) / bench_next_open.replace(0.0, np.nan) - 1.0

    # Stack feature matrices into tidy (date, ticker) DataFrame
    gmap = {t: g for g, mems in groups.items() for t in mems}
    feature_panels = {
        "close_usd": close_w,
        "raw_close_usd": raw_close_w,
        "next_open_usd": next_open_w,
        "f_mom_qual": pf["f_mom_qual"],
        "f_res_mom_vol": pf["f_res_mom_vol"],
        "f_fip": pf["f_fip"],
        "f_ind_mom": pf["f_ind_mom"],
        "f_low_risk": pf["f_low_risk"],
        "f_rev_1w": f_rev_1w,
        "f_mom_3m": f_mom_3m,
        "f_vol_60d": f_vol_60d,
        "f_liq": f_liq,
        "fwd_ret_1w_open": fwd_1w_open,
        **fwd_cols,
    }

    stacked = {name: df.stack(future_stack=True) for name, df in feature_panels.items()}
    out = pd.DataFrame(stacked)
    out.index.names = ["date", "ticker"]
    out = out.reset_index()

    out["market"] = market
    out["is_tw"] = 1.0 if market == "TW" else 0.0
    out["industry"] = out["ticker"].map(lambda t: gmap.get(t, "other"))

    # Attach strictly lagged macro state & benchmark 1w return
    m_join = macro_panel[["macro_stress", "rate_inflation", "term_spread", "vix", "regime"]].copy()
    m_join["bench_1w_open"] = bench_1w_open
    m_join.index.name = "date"
    out = out.merge(m_join.reset_index(), on="date", how="left")

    # Interaction features (capture Macro x Micro regime dynamics even for linear models)
    out["ix_stress_x_lowrisk"] = out["macro_stress"].fillna(0.0) * out["f_low_risk"].fillna(0.0)
    out["ix_stress_x_mom"] = out["macro_stress"].fillna(0.0) * out["f_mom_qual"].fillna(0.0)

    # Drop warmup rows where core momentum & low-risk are both unavailable
    core_ok = out["f_mom_qual"].notna() & out["f_ind_mom"].notna() & out["f_low_risk"].notna()
    out = out.loc[core_ok].reset_index(drop=True)

    # Downcast floats to float32 to keep the 2,000-stock multi-year panel compact & fast on Kaggle
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype(np.float32)

    return out


def build_combined_panel(
    markets: Sequence[str] = ("US", "TW"),
    period: str = "8y",
    horizons: Sequence[int] = (4, 12),
    offline_demo: bool = False,
    demo_stocks_per_market: Optional[int] = None,
    demo_days: int = 1040,
    pit_fundamentals_csv: Optional[Path] = None,
    batch_size: int = 150,
    membership_mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build combined Point-in-Time multi-market training panel (default: US 1,000 + TW 1,000 = 2,000 stocks)."""
    mkt_list = [m.strip().upper() for m in markets if m.strip()]
    for m in mkt_list:
        if m not in MARKET_CONFIG:
            raise ValueError(f"Unsupported market '{m}'. Available: {sorted(MARKET_CONFIG.keys())}")

    macro_d: Optional[pd.DataFrame] = None
    market_frames: List[pd.DataFrame] = []

    for idx_m, mkt in enumerate(mkt_list):
        groups = get_market_groups(mkt)
        tickers = list(dict.fromkeys(t for mems in groups.values() for t in mems))
        bench_sym = REGIONAL_BENCHMARKS.get(mkt, MARKET_CONFIG[mkt]["benchmark"])

        if offline_demo:
            close_d, open_d, vol_d, bench_d, groups = _make_synthetic_market_data(
                mkt,
                groups,
                n_days=demo_days,
                seed=42 + idx_m * 17,
                max_stocks=demo_stocks_per_market,
            )
            raw_close_d = close_d.copy()
            if macro_d is None:
                macro_d = make_offline_macro_data(close_d.index)
        else:
            print(f"\n=== Fetching {mkt} ({len(tickers)} stocks, benchmark={bench_sym}) ===")
            close_d, raw_close_d, open_d, vol_d = _download_batched_ohlcv(tickers, period=period, batch_size=batch_size)
            b_raw = yf.download(bench_sym, period=period, progress=False, auto_adjust=True)
            if b_raw is None or b_raw.empty or "Close" not in b_raw:
                raise RuntimeError(f"Failed to download benchmark {bench_sym} for {mkt}.")
            b_close = b_raw["Close"]
            if isinstance(b_close, pd.DataFrame):
                b_close = b_close.iloc[:, 0]
            bench_d = b_close.dropna()

            # Normalize non-USD markets (e.g., TW) to USD so cross-market prices & returns share one numeraire
            if MARKET_CONFIG[mkt].get("currency", "USD").upper() != "USD":
                print(f"  Normalizing {mkt} prices & benchmark to USD via historical FX...")
                close_d, open_d = convert_prices_to_usd(close_d, open_d, period=period)
                raw_close_d, _ = convert_prices_to_usd(raw_close_d, raw_close_d.copy(), period=period)
                b_df = bench_d.to_frame(name=bench_sym)
                b_usd, _ = convert_prices_to_usd(b_df, b_df.copy(), period=period)
                bench_d = b_usd[bench_sym]

            if macro_d is None:
                print("Downloading lagged macroeconomic regime indicators...")
                macro_d = download_macro_series(period=period)

        frame = _build_single_market_panel(
            market=mkt,
            close_d=close_d,
            open_d=open_d,
            vol_d=vol_d,
            bench_d=bench_d,
            groups=groups,
            macro_d=macro_d,
            horizons=horizons,
            raw_close_d=raw_close_d,
        )
        market_frames.append(frame)

    panel = pd.concat(market_frames, ignore_index=True)

    if membership_mask is not None and not membership_mask.empty:
        from portfolio_sim import _apply_membership_mask
        panel = _apply_membership_mask(panel, membership_mask)

    # Compute cross-sectional ranks (recomputed cleanly after any PIT membership_mask filtering)
    for h in horizons:
        col = f"fwd_vol_adj_{h}w"
        if col in panel.columns:
            panel[f"target_rank_mkt_{h}w"] = (
                panel.groupby(["date", "market"])[col].rank(pct=True).astype(np.float32)
            )
            panel[f"target_rank_global_{h}w"] = (
                panel.groupby("date")[col].rank(pct=True).astype(np.float32)
            )

    if pit_fundamentals_csv is not None:
        panel = _merge_pit_fundamentals(panel, pit_fundamentals_csv)

    panel = panel.sort_values(["date", "market", "ticker"]).reset_index(drop=True)
    return panel


def write_kaggle_dataset_metadata(output_dir: Path, slug: str, title: str) -> Path:
    """Write `dataset-metadata.json` so the user can upload to Kaggle in one CLI command."""
    meta = {
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = output_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Point-in-Time US+TW (2,000-stock) Training Panel for Kaggle")
    parser.add_argument("--markets", default="US,TW", help="Comma-separated markets (default: US,TW = 2,000 stocks)")
    parser.add_argument("--period", default="8y", help="Yahoo Finance history window (default: 8y)")
    parser.add_argument("--horizons", default="4,12", help="Forward target horizons in weeks (default: 4,12)")
    parser.add_argument("--batch-size", type=int, default=150, help="Yahoo Finance download chunk size (default: 150)")
    parser.add_argument("--pit-fundamentals-csv", type=Path, default=None, help="Optional Point-in-Time historical fundamentals CSV")
    parser.add_argument("--membership-csv", type=Path, default=None, help="Optional point-in-time constituent mask CSV")
    parser.add_argument("--output", type=Path, default=Path("kaggle_dataset/snowball_us_tw_2000.parquet"), help="Output file path (.parquet or .csv.gz)")
    parser.add_argument("--kaggle-slug", default="", help="Optional '<username>/snowball-us-tw-2000' to emit dataset-metadata.json")
    parser.add_argument("--offline-demo", action="store_true", help="Build synthetic 2,000-stock panel offline for testing")
    parser.add_argument("--demo-stocks-per-market", type=int, default=None, help="Optional cap per market in offline demo mode")
    args = parser.parse_args()

    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())
    mem_mask = pd.read_csv(args.membership_csv, index_col=0, parse_dates=True).astype(bool) if args.membership_csv else None

    panel = build_combined_panel(
        markets=markets,
        period=args.period,
        horizons=horizons,
        offline_demo=args.offline_demo,
        demo_stocks_per_market=args.demo_stocks_per_market,
        pit_fundamentals_csv=args.pit_fundamentals_csv,
        batch_size=args.batch_size,
        membership_mask=mem_mask,
    )
    saved_path = save_dataframe(panel, args.output)

    if args.kaggle_slug.strip():
        meta_path = write_kaggle_dataset_metadata(
            saved_path.parent,
            slug=args.kaggle_slug.strip(),
            title="Snowball US+TW 2000-Stock Point-in-Time Multi-Factor Panel",
        )
        print(f"Saved Kaggle dataset metadata: {meta_path}")

    n_tickers = panel["ticker"].nunique()
    by_mkt = panel.groupby("market")["ticker"].nunique().to_dict()
    print(f"\n=== Training Panel Built Successfully ===")
    print(f"  File       : {saved_path}")
    print(f"  Rows       : {len(panel):,}")
    print(f"  Tickers    : {n_tickers:,} ({by_mkt})")
    print(f"  Date Range : {panel['date'].min().date()} -> {panel['date'].max().date()} ({panel['date'].nunique()} weeks)")
    print(f"  Columns    : {len(panel.columns)}")


if __name__ == "__main__":
    main()
