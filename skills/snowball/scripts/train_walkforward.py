"""Purged Expanding Walk-Forward Multi-Market Training Engine (US + TW 2,000 Stocks).

Designed to run seamlessly both on Kaggle (GPU/CPU) and locally with zero look-ahead bias:

1. Strict Purge Gap:
   For every refit date `t_refit`, training samples are strictly restricted to:
       `date <= t_refit - timedelta(weeks=horizon)`
   guaranteeing every training label (`fwd_ret_{horizon}w`) has completely settled before `t_refit`.

2. 4-Model OOS Benchmark Comparison (identical OOS dates & universe):
   - `baseline_v6_prior`     : Zero-parameter theory prior (macro-conditioned price pillars)
   - `stage1_icir_shrinkage` : Bayesian Shrinkage ICIR rolling weights (50% prior + 50% settled ICIR)
   - `stage2_ridge`          : Cross-sectional L2-regularized Ridge regression (pure NumPy, zero extra deps)
   - `stage3_lgbm`           : Shallow regularized LightGBM (`max_depth=3`, `num_leaves=7`, `reg_lambda=25`)
                               with automatic GPU/CPU detection (skipped gracefully if lightgbm absent)

3. Institutional OOS Evaluation:
   - Global, US-only, and TW-only Spearman Rank IC, ICIR, Newey-West t-statistic, IC Hit Rate
   - Walk-Forward Platt/Logistic calibrated Brier Skill Score (BSS vs coin flip)
   - Realistic Top-K weekly portfolio simulation using Next-Trading-Day Open returns & turnover costs
     (TWR CAGR, TWR Sharpe, TWR Max Drawdown, Average Weekly Turnover)

4. Production Artifact Export:
   - `v6_learned_weights.json` (directly loadable by `global_ranker.py --learned-weights`)
   - `lgbm_model.txt` (if LightGBM is installed)
   - `oos_summary.json`, `weight_history.csv`, `oos_predictions.parquet` (or `.csv.gz`)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate as ev
from build_training_panel import (
    CORE_FEATURE_COLS,
    PIT_FUNDAMENTAL_COLS,
    build_combined_panel,
    load_dataframe,
    save_dataframe,
)
from macro_micro import BASELINE_V6_WEIGHTS, WEIGHT_BOUNDS

# Price-available pillars present in every historical panel row (before optional PIT fundamentals)
PILLAR_PRIOR_WEIGHTS: Dict[str, float] = {
    "f_mom_qual": 0.45,
    "f_ind_mom": 0.25,
    "f_low_risk": 0.20,
    "f_rev_1w": 0.10,
}

# Mapping from price-only trained pillar weights back into full 6-pillar v6 weights (preserving valuation/quality/fund budget)
PRICE_TO_V6_MAP: Dict[str, str] = {
    "f_mom_qual": "mom_qual",
    "f_ind_mom": "ind_mom",
    "f_low_risk": "low_risk",
}


def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


def detect_feature_columns(panel: pd.DataFrame) -> List[str]:
    """Return available numeric feature columns (core price/macro + any present PIT fundamentals)."""
    candidates = list(CORE_FEATURE_COLS) + list(PIT_FUNDAMENTAL_COLS)
    return [c for c in candidates if c in panel.columns and panel[c].notna().any()]


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return np.nan
    ax, bx = a[mask], b[mask]
    if np.nanstd(ax) <= 1e-12 or np.nanstd(bx) <= 1e-12:
        return np.nan
    return float(spearmanr(ax, bx).statistic)


def _compute_daily_ic_series(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    market_filter: Optional[str] = None,
) -> pd.Series:
    sub = df if market_filter is None else df.loc[df["market"] == market_filter]
    vals: Dict[pd.Timestamp, float] = {}
    for dt, grp in sub.groupby("date", sort=True):
        ic = _safe_spearman(grp[score_col].to_numpy(dtype=float), grp[target_col].to_numpy(dtype=float))
        if np.isfinite(ic):
            vals[dt] = ic
    return pd.Series(vals, dtype=float).sort_index()


def _fit_icir_shrinkage_weights(
    train_df: pd.DataFrame,
    target_col: str,
    prior: Dict[str, float] = PILLAR_PRIOR_WEIGHTS,
    shrink: float = 0.50,
    min_weeks: int = 16,
) -> Dict[str, float]:
    """Bayesian Shrinkage ICIR weights: w = (1 - shrink) * prior + shrink * norm(max(ICIR, 0))."""
    p_tot = sum(prior.values())
    base = {k: v / p_tot for k, v in prior.items()}

    icir_pos: Dict[str, float] = {}
    for col in prior:
        if col not in train_df.columns:
            icir_pos[col] = 0.0
            continue
        ic_s = _compute_daily_ic_series(train_df, col, target_col)
        if len(ic_s) < min_weeks or ic_s.std() <= 1e-12:
            icir_pos[col] = 0.0
        else:
            icir_pos[col] = max(float(ic_s.mean() / ic_s.std()), 0.0)

    tot = sum(icir_pos.values())
    if tot <= 1e-12:
        return base
    return {k: (1.0 - shrink) * base[k] + shrink * (icir_pos[k] / tot) for k in prior}


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    l2_alpha: float = 500.0,
) -> Tuple[float, np.ndarray]:
    """Exact closed-form standardized L2-regularized Ridge regression (pure NumPy, fast & stable)."""
    # Replace NaNs with 0 (features are already cross-sectional z-scores centered at 0)
    Xf = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float64)
    yf = y.astype(np.float64)
    y_mean = float(np.mean(yf))
    yc = yf - y_mean
    n_features = Xf.shape[1]
    # Scale alpha with sample count so regularization strength stays consistent as expanding window grows
    eff_alpha = l2_alpha * max(len(yf) / 20000.0, 0.25)
    gram = Xf.T @ Xf + eff_alpha * np.eye(n_features, dtype=np.float64)
    rhs = Xf.T @ yc
    coef = np.linalg.solve(gram, rhs)
    return y_mean, coef


def _predict_ridge(X: np.ndarray, intercept: float, coef: np.ndarray) -> np.ndarray:
    Xf = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float64)
    return intercept + Xf @ coef


def _fit_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    use_gpu: bool = False,
    seed: int = 42,
):
    """Fit shallow, strongly regularized LightGBM Huber regressor designed for noisy equity cross-sections."""
    import lightgbm as lgb

    params = {
        "objective": "huber",
        "alpha": 0.25,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": max(50, min(200, len(y_train) // 50)),
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 5.0,
        "reg_lambda": 25.0,
        "n_estimators": 220,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    if use_gpu:
        params["device"] = "gpu"

    model = lgb.LGBMRegressor(**params)
    X_df = pd.DataFrame(X_train, columns=list(feature_names))
    try:
        model.fit(X_df, y_train)
    except Exception:
        # Automatic graceful fallback to CPU if Kaggle GPU OpenCL driver unavailable
        params.pop("device", None)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_df, y_train)
    return model


def _baseline_prior_score(df: pd.DataFrame) -> np.ndarray:
    """Zero-parameter macro-conditioned baseline score matching Snowball v6 price pillars."""
    st = df["macro_stress"].fillna(0.0).clip(-1.5, 1.5).to_numpy(dtype=float)
    w_mom = np.clip(0.45 - 0.10 * st, 0.25, 0.55)
    w_ind = np.clip(0.25 - 0.05 * st, 0.15, 0.30)
    w_def = np.clip(0.20 + 0.15 * st, 0.08, 0.38)
    w_rev = 0.10
    return (
        w_mom * df["f_mom_qual"].fillna(0.0).to_numpy(dtype=float)
        + w_ind * df["f_ind_mom"].fillna(0.0).to_numpy(dtype=float)
        + w_def * df["f_low_risk"].fillna(0.0).to_numpy(dtype=float)
        + w_rev * df["f_rev_1w"].fillna(0.0).to_numpy(dtype=float)
    )


def _simulate_topk_twr(
    oos_df: pd.DataFrame,
    score_col: str,
    top_k: int = 40,
    rebalance_every: int = 4,
    round_trip_cost: float = 0.0025,
) -> Dict[str, float]:
    """4-cohort staggered Top-K long-only weekly portfolio using Next-Day Open 1-week returns."""
    if "fwd_ret_1w_open" not in oos_df.columns:
        return {"TWR_CAGR": np.nan, "TWR_Sharpe": np.nan, "TWR_MDD": np.nan, "Avg_Weekly_Turnover": np.nan}

    dates = sorted(oos_df["date"].dropna().unique())
    if len(dates) < 8:
        return {"TWR_CAGR": np.nan, "TWR_Sharpe": np.nan, "TWR_MDD": np.nan, "Avg_Weekly_Turnover": np.nan}

    by_date = {dt: grp for dt, grp in oos_df.groupby("date")}
    cohort_series: List[pd.Series] = []
    turnover_list: List[float] = []

    for offset in range(rebalance_every):
        held: set = set()
        wk_ret: Dict[pd.Timestamp, float] = {}
        for i, dt in enumerate(dates):
            grp = by_date[dt]
            cost = 0.0
            if i % rebalance_every == offset or not held:
                ranked = grp.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
                k = min(top_k, max(4, len(ranked) // 10))
                new_set = set(ranked["ticker"].iloc[:k].tolist())
                if held:
                    turn = len(new_set - held) / max(len(new_set), 1)
                    cost = turn * round_trip_cost
                    turnover_list.append(turn)
                held = new_set

            sub = grp.loc[grp["ticker"].isin(held), "fwd_ret_1w_open"].dropna()
            gross = float(sub.mean()) if len(sub) > 0 else 0.0
            wk_ret[dt] = gross - cost
        cohort_series.append(pd.Series(wk_ret))

    port_w = pd.concat(cohort_series, axis=1).mean(axis=1).dropna()
    if len(port_w) < 4:
        return {"TWR_CAGR": np.nan, "TWR_Sharpe": np.nan, "TWR_MDD": np.nan, "Avg_Weekly_Turnover": np.nan}

    nav = (1.0 + port_w).cumprod()
    n_w = len(port_w)
    cagr = float(nav.iloc[-1] ** (52.0 / n_w) - 1.0)
    mdd = float((nav / nav.cummax() - 1.0).min())
    sd = float(port_w.std())
    sharpe = float(port_w.mean() / sd * np.sqrt(52.0)) if sd > 1e-12 else np.nan
    avg_turn = float(np.mean(turnover_list)) / rebalance_every if turnover_list else 0.0
    return {
        "TWR_CAGR": cagr,
        "TWR_Sharpe": sharpe,
        "TWR_MDD": mdd,
        "Avg_Weekly_Turnover": avg_turn,
    }


def _evaluate_model_oos(
    oos_df: pd.DataFrame,
    score_col: str,
    prob_col: str,
    target_col: str,
    excess_col: str,
    horizon: int,
    top_k: int = 40,
) -> Dict[str, float]:
    """Compute comprehensive institutional OOS metrics (Global + US + TW + BSS + Net TWR)."""
    ic_all = _compute_daily_ic_series(oos_df, score_col, target_col)
    ic_us = _compute_daily_ic_series(oos_df, score_col, target_col, market_filter="US")
    ic_tw = _compute_daily_ic_series(oos_df, score_col, target_col, market_filter="TW")

    ic_mean = float(ic_all.mean()) if len(ic_all) else np.nan
    ic_std = float(ic_all.std()) if len(ic_all) > 1 else np.nan
    icir = float(ic_mean / ic_std) if (np.isfinite(ic_std) and ic_std > 1e-12) else np.nan
    t_nw = float(ev.newey_west_t(ic_all, lags=max(4, horizon))) if len(ic_all) >= 6 else np.nan
    hit = float((ic_all > 0).mean()) if len(ic_all) else np.nan

    # Brier Skill Score on binary outperformance event (fwd_excess > 0)
    valid_p = oos_df[[prob_col, excess_col]].dropna()
    if len(valid_p) >= 50:
        p = valid_p[prob_col].to_numpy(dtype=float)
        o = (valid_p[excess_col].to_numpy(dtype=float) > 0.0).astype(float)
        brier_val = ev.brier(p, o)
        bss_val = ev.brier_skill(p, o)
    else:
        brier_val, bss_val = np.nan, np.nan

    twr = _simulate_topk_twr(oos_df, score_col, top_k=top_k, rebalance_every=max(1, min(horizon, 4)))

    return {
        "OOS_Weeks": int(len(ic_all)),
        "Rank_IC": ic_mean,
        "ICIR": icir,
        "t_NW": t_nw,
        "IC_Hit_Rate": hit,
        "US_Rank_IC": float(ic_us.mean()) if len(ic_us) else np.nan,
        "TW_Rank_IC": float(ic_tw.mean()) if len(ic_tw) else np.nan,
        "Brier": brier_val,
        "BSS": bss_val,
        **twr,
    }


def map_learned_weights_to_v6(
    icir_weights: Dict[str, float],
    ridge_coef: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Translate trained price-pillar ICIR/Ridge weights into bounded 6-pillar Snowball v6 weights.

    Preserves the institutional budget for fundamental pillars (`norm_val`, `quality`, `fund`)
    while adapting the price-pillar allocation (`mom_qual`, `ind_mom`, `low_risk`) using
    learned OOS signal strength, bounded strictly inside `WEIGHT_BOUNDS`.
    """
    out = dict(BASELINE_V6_WEIGHTS)
    price_budget = out["mom_qual"] + out["ind_mom"] + out["low_risk"]

    raw_price = {}
    for src, dst in PRICE_TO_V6_MAP.items():
        w_s1 = max(float(icir_weights.get(src, 0.0)), 0.0)
        if ridge_coef and src in ridge_coef:
            w_s2 = max(float(ridge_coef[src]), 0.0)
            raw_price[dst] = 0.65 * w_s1 + 0.35 * w_s2
        else:
            raw_price[dst] = w_s1

    tot_p = sum(raw_price.values())
    if tot_p > 1e-12:
        for dst, val in raw_price.items():
            lo, hi = WEIGHT_BOUNDS[dst]
            out[dst] = float(np.clip((val / tot_p) * price_budget, lo, hi))

    # Re-normalize all 6 pillars to sum to exactly 1.00
    s_all = sum(out.values())
    return {k: round(v / s_all, 4) for k, v in out.items()}


def run_purged_walkforward(
    panel: pd.DataFrame,
    horizon: int = 4,
    min_train_weeks: int = 52,
    refit_step_weeks: int = 8,
    ridge_alpha: float = 500.0,
    enable_lgbm: bool = True,
    use_gpu: bool = False,
    top_k: int = 40,
    seed: int = 42,
) -> Dict[str, object]:
    """Execute strict Purged Expanding Walk-Forward training across all 4 models."""
    target_col = f"target_rank_{horizon}w"
    excess_col = f"fwd_excess_{horizon}w"
    if target_col not in panel.columns:
        raise ValueError(f"Target column '{target_col}' not found in panel.")

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "market", "ticker"]).reset_index(drop=True)

    feature_cols = detect_feature_columns(df)
    all_dates = np.sort(df["date"].unique())
    if len(all_dates) <= min_train_weeks + horizon + 4:
        raise ValueError(
            f"Not enough weekly dates ({len(all_dates)}) for min_train_weeks={min_train_weeks} + horizon={horizon}."
        )

    oos_start_idx = min_train_weeks + horizon
    oos_dates = all_dates[oos_start_idx:]

    lgbm_active = enable_lgbm and _has_lightgbm()
    models_to_run = ["baseline_v6_prior", "stage1_icir_shrinkage", "stage2_ridge"]
    if lgbm_active:
        models_to_run.append("stage3_lgbm")

    oos_frames: List[pd.DataFrame] = []
    weight_history: List[dict] = []
    purge_audit: List[dict] = []
    last_icir_w = dict(PILLAR_PRIOR_WEIGHTS)
    last_ridge_map: Dict[str, float] = {}
    last_lgbm_model = None
    lgbm_importances: List[np.ndarray] = []

    for block_start in range(0, len(oos_dates), refit_step_weeks):
        block_dates = oos_dates[block_start : block_start + refit_step_weeks]
        refit_date = pd.Timestamp(block_dates[0])
        # STRICT PURGE GAP: training sample date + horizon weeks <= refit_date
        max_settled_date = refit_date - pd.Timedelta(weeks=horizon)

        train_mask = (df["date"] <= max_settled_date) & df[target_col].notna()
        test_mask = df["date"].isin(block_dates)
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask].copy()
        if len(train_df) < 200 or test_df.empty:
            continue

        # Assert strict zero-overlap purge guarantee
        actual_max_train = pd.Timestamp(train_df["date"].max())
        gap_days = (refit_date - actual_max_train).days
        if gap_days < horizon * 7:
            raise AssertionError(
                f"Purge gap violation! max_train={actual_max_train.date()}, refit={refit_date.date()}, gap={gap_days}d < {horizon * 7}d"
            )
        purge_audit.append({
            "refit_date": refit_date.date().isoformat(),
            "max_train_date": actual_max_train.date().isoformat(),
            "gap_days": int(gap_days),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        })

        # 1. Baseline (Zero Parameters)
        train_base = _baseline_prior_score(train_df)
        test_df["pred_baseline_v6_prior"] = _baseline_prior_score(test_df)

        # 2. Stage 1: Bayesian Shrinkage ICIR
        w_icir = _fit_icir_shrinkage_weights(train_df, target_col=target_col)
        last_icir_w = w_icir
        train_s1 = sum(w_icir[c] * train_df[c].fillna(0.0).to_numpy(dtype=float) for c in w_icir if c in train_df.columns)
        test_s1 = sum(w_icir[c] * test_df[c].fillna(0.0).to_numpy(dtype=float) for c in w_icir if c in test_df.columns)
        test_df["pred_stage1_icir_shrinkage"] = test_s1

        # 3. Stage 2: Cross-Sectional L2 Ridge
        X_tr = train_df[feature_cols].to_numpy(dtype=float)
        y_tr = train_df[target_col].to_numpy(dtype=float)
        X_te = test_df[feature_cols].to_numpy(dtype=float)
        r_int, r_coef = _fit_ridge(X_tr, y_tr, l2_alpha=ridge_alpha)
        last_ridge_map = {col: float(val) for col, val in zip(feature_cols, r_coef)}
        train_s2 = _predict_ridge(X_tr, r_int, r_coef)
        test_df["pred_stage2_ridge"] = _predict_ridge(X_te, r_int, r_coef)

        # 4. Stage 3: Shallow Regularized LightGBM (when installed, e.g. on Kaggle)
        train_preds_map = {
            "baseline_v6_prior": train_base,
            "stage1_icir_shrinkage": train_s1,
            "stage2_ridge": train_s2,
        }
        if lgbm_active:
            lgb_model = _fit_lightgbm(X_tr, y_tr, feature_cols, use_gpu=use_gpu, seed=seed)
            last_lgbm_model = lgb_model
            lgbm_importances.append(np.asarray(lgb_model.feature_importances_, dtype=float))
            X_tr_df = pd.DataFrame(X_tr, columns=feature_cols)
            X_te_df = pd.DataFrame(X_te, columns=feature_cols)
            train_preds_map["stage3_lgbm"] = lgb_model.predict(X_tr_df)
            test_df["pred_stage3_lgbm"] = lgb_model.predict(X_te_df)

        # Walk-forward Platt/Logistic probability calibration on settled training window only
        y_bin_tr = (train_df[excess_col].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float)
        for m_name in models_to_run:
            tr_sig = np.asarray(train_preds_map[m_name], dtype=float)
            sd_tr = float(np.nanstd(tr_sig))
            mu_tr = float(np.nanmean(tr_sig))
            if sd_tr > 1e-12 and len(tr_sig) >= 200:
                tr_z = np.clip((tr_sig - mu_tr) / sd_tr, -4.0, 4.0)
                a, k = ev.fit_logistic(tr_z, y_bin_tr)
                te_z = np.clip((test_df[f"pred_{m_name}"].to_numpy(dtype=float) - mu_tr) / sd_tr, -4.0, 4.0)
                test_df[f"prob_{m_name}"] = 1.0 / (1.0 + np.exp(-(a + k * te_z)))
            else:
                test_df[f"prob_{m_name}"] = 0.50

        weight_history.append({
            "refit_date": refit_date.date().isoformat(),
            **{f"w_{k}": round(v, 4) for k, v in w_icir.items()},
            **{f"ridge_{k}": round(last_ridge_map.get(k, 0.0), 5) for k in PILLAR_PRIOR_WEIGHTS},
        })
        oos_frames.append(test_df)

    if not oos_frames:
        raise RuntimeError("No OOS walk-forward blocks produced. Check panel date span and min_train_weeks.")

    oos_df = pd.concat(oos_frames, ignore_index=True)

    # Evaluate every model on identical OOS panel
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    for m_name in models_to_run:
        metrics_by_model[m_name] = _evaluate_model_oos(
            oos_df=oos_df,
            score_col=f"pred_{m_name}",
            prob_col=f"prob_{m_name}",
            target_col=target_col,
            excess_col=excess_col,
            horizon=horizon,
            top_k=top_k,
        )

    # Fit final production LightGBM on all fully settled data (latest date - horizon)
    final_lgbm_model = last_lgbm_model
    if lgbm_active:
        latest_settled = pd.Timestamp(all_dates[-1]) - pd.Timedelta(weeks=horizon)
        final_train = df.loc[(df["date"] <= latest_settled) & df[target_col].notna()]
        if len(final_train) >= 200:
            final_lgbm_model = _fit_lightgbm(
                final_train[feature_cols].to_numpy(dtype=float),
                final_train[target_col].to_numpy(dtype=float),
                feature_cols,
                use_gpu=use_gpu,
                seed=seed,
            )

    learned_v6_weights = map_learned_weights_to_v6(last_icir_w, last_ridge_map)
    feat_imp = (
        {col: round(float(val), 2) for col, val in zip(feature_cols, np.mean(lgbm_importances, axis=0))}
        if lgbm_importances
        else {}
    )

    return {
        "metrics": metrics_by_model,
        "oos_predictions": oos_df,
        "weight_history": pd.DataFrame(weight_history),
        "purge_audit": purge_audit,
        "feature_cols": feature_cols,
        "lgbm_feature_importance": feat_imp,
        "learned_v6_weights": learned_v6_weights,
        "final_lgbm_model": final_lgbm_model,
        "config": {
            "horizon_weeks": horizon,
            "min_train_weeks": min_train_weeks,
            "refit_step_weeks": refit_step_weeks,
            "ridge_alpha": ridge_alpha,
            "lgbm_active": lgbm_active,
            "n_refits": len(purge_audit),
            "oos_start": str(pd.Timestamp(oos_df["date"].min()).date()),
            "oos_end": str(pd.Timestamp(oos_df["date"].max()).date()),
            "markets": sorted(oos_df["market"].unique().tolist()),
            "n_tickers": int(oos_df["ticker"].nunique()),
        },
    }


def export_training_artifacts(result: Dict[str, object], output_dir: Path) -> Dict[str, str]:
    """Persist portable model artifacts & OOS diagnostics for production inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, str] = {}

    # 1. Portable v6 learned weights JSON (directly usable by global_ranker.py --learned-weights)
    weights_payload = {
        "schema": "snowball_v6_learned_weights_v1",
        "weights": result["learned_v6_weights"],
        "config": result["config"],
        "oos_metrics": result["metrics"],
    }
    w_path = output_dir / "v6_learned_weights.json"
    w_path.write_text(json.dumps(weights_payload, indent=2, default=str), encoding="utf-8")
    exported["v6_learned_weights"] = str(w_path)

    # 2. Full OOS summary + purge audit + feature importance
    summary_payload = {
        "config": result["config"],
        "metrics": result["metrics"],
        "learned_v6_weights": result["learned_v6_weights"],
        "lgbm_feature_importance": result["lgbm_feature_importance"],
        "purge_audit": result["purge_audit"],
    }
    s_path = output_dir / "oos_summary.json"
    s_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    exported["oos_summary"] = str(s_path)

    # 3. Rolling weight evolution CSV
    wh: pd.DataFrame = result["weight_history"]
    wh_path = output_dir / "weight_history.csv"
    wh.to_csv(wh_path, index=False)
    exported["weight_history"] = str(wh_path)

    # 4. OOS predictions panel
    pred_cols = [
        c for c in result["oos_predictions"].columns
        if c in ("date", "ticker", "market", "industry", "fwd_ret_1w_open")
        or c.startswith(("pred_", "prob_", "target_rank_", "fwd_excess_"))
    ]
    pred_path = save_dataframe(result["oos_predictions"][pred_cols], output_dir / "oos_predictions.parquet")
    exported["oos_predictions"] = str(pred_path)

    # 5. Final LightGBM booster file (if trained)
    lgbm_model = result.get("final_lgbm_model")
    if lgbm_model is not None:
        m_path = output_dir / "lgbm_model.txt"
        lgbm_model.booster_.save_model(str(m_path))
        exported["lgbm_model"] = str(m_path)

    return exported


def format_metrics_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, m in metrics.items():
        rows.append({"model": model_name, **m})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball Purged Walk-Forward Training Engine (US+TW 2,000 Stocks)")
    parser.add_argument("--panel", type=Path, default=None, help="Path to pre-built panel (.parquet or .csv.gz)")
    parser.add_argument("--markets", default="US,TW", help="Markets if building offline demo on the fly (default: US,TW)")
    parser.add_argument("--horizon", type=int, default=4, help="Forward prediction horizon in weeks (default: 4)")
    parser.add_argument("--min-train-weeks", type=int, default=52, help="Minimum expanding warmup weeks (default: 52)")
    parser.add_argument("--refit-step", type=int, default=8, help="Refit cadence in weeks (default: 8)")
    parser.add_argument("--ridge-alpha", type=float, default=500.0, help="Ridge L2 regularization strength (default: 500)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K portfolio size for OOS TWR simulation (default: 40)")
    parser.add_argument("--no-lgbm", action="store_true", help="Disable Stage 3 LightGBM even if installed")
    parser.add_argument("--gpu", action="store_true", help="Enable LightGBM GPU acceleration (Kaggle T4/P100)")
    parser.add_argument("--output-dir", type=Path, default=Path("snowball_artifacts"), help="Directory for exported artifacts")
    parser.add_argument("--offline-demo", action="store_true", help="Run end-to-end on synthetic US+TW data without network")
    parser.add_argument("--demo-stocks-per-market", type=int, default=60, help="Stocks per market in fast offline demo (default: 60)")
    args = parser.parse_args()

    if args.panel is not None:
        panel = load_dataframe(args.panel)
    elif args.offline_demo:
        markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
        panel = build_combined_panel(
            markets=markets,
            horizons=(args.horizon,),
            offline_demo=True,
            demo_stocks_per_market=args.demo_stocks_per_market,
        )
    else:
        raise SystemExit("Provide --panel path/to/snowball_us_tw_2000.parquet or pass --offline-demo.")

    res = run_purged_walkforward(
        panel=panel,
        horizon=args.horizon,
        min_train_weeks=args.min_train_weeks,
        refit_step_weeks=args.refit_step,
        ridge_alpha=args.ridge_alpha,
        enable_lgbm=not args.no_lgbm,
        use_gpu=args.gpu,
        top_k=args.top_k,
    )
    exported = export_training_artifacts(res, args.output_dir)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== Snowball Purged Walk-Forward OOS Comparison ===")
    print(f"Config: {res['config']}")
    tbl = format_metrics_table(res["metrics"])
    print(tbl.round(4).to_string(index=False))
    print(f"\nExported v6 learned weights: {res['learned_v6_weights']}")
    print("Artifacts:")
    for k, v in exported.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
