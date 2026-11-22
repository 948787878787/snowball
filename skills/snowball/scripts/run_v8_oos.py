"""One-shot historical walk-forward evaluation of the frozen Snowball v8 config on the PIT panel.

Runs only when `configs/v8_frozen.json` is FROZEN with a code commit (pre-registration). Every reported
strategy line is appended to `results/trial_log.csv`; DSR uses N = trial rows after logging + 20.

  ridge arm : sign-constrained Ridge refit every `refit_every_weeks` on labels settled before the refit
              date (embargo), first refit after `min_train_weeks` settled weeks; scores the next block.
  arms      : P1 = 0.5 rank(v8_zero) + 0.5 rank(v8_ridge); P2 = v6 prior; P3 = v8_zero; industry-demeaned.
  standard  : v7 Stage 3 engine (top 20, 4 cohorts, 4-week rebalance, buffer 10, industry cap 25%).
  user rules: 100,000 TWD lump, 10,000 per weekly buy, odd lots + slippage, zero realized loss.

Usage (after freezing):
  python run_v8_oos.py --panel ../../../../kaggle_dataset/v8/snowball_tw_v8.parquet \
      --config ../../../configs/v8_frozen.json --price-wide ../../../../kaggle_dataset/v8/tw_pit_next_open_usd_wide.parquet \
      --trial-log ../../../results/trial_log.csv --out-dir ../../../../snowball_artifacts_tw_v8_oos
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import cashflow_sim as cfs
import run_cashflow_sim as rcs
import evaluate as ev
import portfolio_sim as psim
import stage5_validation as s5
import train_walkforward as twf
import v8_composite as vc
import v8_eval_metrics as em
import v8_model as vm
from paper_trade import ARM_SCORES

HOLDOUT_CUTOFF = pd.Timestamp("2025-11-20")
EXTRA_TRIALS = 20
STANDARD_KW: Dict[str, Any] = dict(top_k=20, rebalance_every=4, market="TW", max_industry_weight=0.25,
                   buffer_mult=10.0, neutralize_scores=False)
USER_VARIANTS = [("P1_v8_blend", 5), ("P1_v8_blend", 10), ("P1_v8_blend", None),
                 ("P2_v6_prior", None), ("P3_v8_zero", None)]


def require_frozen(cfg: Dict[str, Any]) -> None:
    if cfg.get("status") != "FROZEN" or not cfg.get("code_commit"):
        raise PermissionError("v8 historical OOS runs only on a FROZEN config with code_commit set")


def walkforward_ridge(panel: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add `v8_ridge`, `model_train_end`, `model_label_last`; NaN before the first refit."""
    tc = cfg["trained"]
    df = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    X, names, bounds = vm.design_matrix(df, cfg)
    dates = sorted(df["date"].unique())
    first = int(tc.get("min_train_weeks", 104)) + int(tc["label_embargo_weeks"])
    step = int(tc["refit_every_weeks"])
    pred = np.full(len(df), np.nan)
    train_end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    label_last = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for i in range(first, len(dates), step):
        r = pd.Timestamp(dates[i])
        block_end = pd.Timestamp(dates[min(i + step, len(dates)) - 1])
        model = vm.fit_design(X, names, bounds, df, cfg, train_end=r)
        rows = ((df["date"] >= r) & (df["date"] <= block_end)).to_numpy()
        preds = [model["intercept"][t] + X[rows] @ np.asarray(model["coef"][t]) for t in tc["targets"]]
        pred[rows] = np.mean(preds, axis=0)
        train_end[rows] = r
        label_last[rows] = pd.Timestamp(model["train_last_date"])
    return df.assign(v8_ridge=pred, model_train_end=train_end, model_label_last=label_last)


def arm_scores(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    ind = cfg.get("industry_col", "industry")
    out = df.copy()
    for arm, parts in ARM_SCORES.items():
        if arm in cfg["arms"]:
            out[f"{arm}_raw"] = vc.blend_ranks(out, parts)
            out[f"{arm}_score"] = vc.industry_demean(out, f"{arm}_raw", industry_col=ind)
    return out


def _to_usd_returns(twd_px: pd.Series, fx: pd.Series) -> pd.Series:
    """Weekly execution-to-execution return of a TWD instrument expressed in USD, indexed by signal date."""
    r_twd = twd_px.shift(-1) / twd_px - 1.0
    return ((1.0 + r_twd) * (fx / fx.shift(-1)) - 1.0).dropna()


def standard_eval(sub: pd.DataFrame, score: str, ew_w: pd.Series, etf_w: pd.Series, n_trials: int) -> Dict[str, Any]:
    sub = sub.assign(industry=sub["industry_twse"])
    board = psim.simulate_topk_twr(sub, score, tw_odd_lot=False, **STANDARD_KW)
    odd = psim.simulate_topk_twr(sub, score, tw_odd_lot=True, **STANDARD_KW)
    w_board = s5.compute_portfolio_weekly_returns(sub, score, tw_odd_lot=False, **STANDARD_KW)
    w_odd = s5.compute_portfolio_weekly_returns(sub, score, tw_odd_lot=True, **STANDARD_KW)
    ic = twf._compute_daily_ic_series(sub, score, "target_rank_4w", market_filter="TW")
    ex = s5.compute_excess_metrics(w_board, ew_w.loc[ew_w.index.isin(w_board.index)], lags=4)
    ex_odd = s5.compute_excess_metrics(w_odd, ew_w.loc[ew_w.index.isin(w_odd.index)], lags=4)
    vs_etf = s5.compute_excess_metrics(w_board, etf_w.loc[etf_w.index.isin(w_board.index)], lags=4)
    dsr = ev.deflated_sharpe_ratio(board["TWR_Sharpe"], n_trials=n_trials, n_obs=len(w_board))
    return {
        "n_weeks": len(w_board), "IC": float(ic.mean()), "IC_tNW": float(ev.newey_west_t(ic, lags=4)),
        "Board_CAGR": board["TWR_CAGR"], "Board_Sharpe": board["TWR_Sharpe"], "MDD": board["TWR_MDD"],
        "Odd_CAGR": odd["TWR_CAGR"], "Odd_MDD": odd["TWR_MDD"], "Rebalance_Turnover": board["Avg_Rebalance_Turnover"],
        **ex, "Odd_Net_Excess_CAGR": ex_odd["Net_Excess_CAGR"], "Odd_Excess_t_NW": ex_odd["Excess_t_NW"],
        "Excess_t_NW_vs_0050": vs_etf["Excess_t_NW"], "DSR_cons": dsr,
        **{f"TUW_{k}": v for k, v in em.time_under_water((1.0 + w_board).cumprod()).items()},
        **{f"Hurdle3y_odd_{k}": v for k, v in em.rolling_hurdle_windows(w_odd).items()},
        "Pass_Gate": s5.passes_v7_relative_gate(ex["Excess_t_NW"], dsr),
        "Harvey_t3": bool(np.isfinite(ex["Excess_t_NW"]) and ex["Excess_t_NW"] >= 3.0),
    }


def user_rules_config(cfg: Dict[str, Any], max_names: Optional[int]) -> cfs.CashflowConfig:
    ur, co = cfg["portfolios"]["user_rules"], cfg["costs"]
    return cfs.CashflowConfig(
        monthly_cash=0.0, initial_cash=float(ur["initial_capital_twd"]), buy_amount=float(ur["buy_amount_twd"]),
        n_buy=int(ur["n_buy_per_week"]), buy_pct=float(ur["buy_pct"]), sell_pct=float(ur["sell_pct"]),
        fee_rate=float(co["fee_rate"]), min_fee=float(co["min_fee_twd"]), sell_tax=float(co["sell_tax"]),
        max_weight=float(ur["max_weight"]), allow_loss_sells=not ur["zero_realized_loss"],
        max_names=max_names, industry_cap=ur.get("industry_cap"), slippage=float(ur.get("odd_lot_slippage", 0.0)),
    )


def user_rules_eval(df: pd.DataFrame, arm: str, ccfg: cfs.CashflowConfig, fallback: Optional[pd.DataFrame],
                    etf_twd: pd.Series, hurdle: float) -> Dict[str, Any]:
    res = cfs.simulate(df, f"{arm}_score", "px_twd", ccfg, delist_haircut=0.0,
                       price_fallback=fallback, industry_col="industry_twse")
    twr = cfs.weekly_twr(res.values, res.contributions)
    etf_v, etf_c = cfs.lump_benchmark(etf_twd.loc[etf_twd.index >= res.values.index[0]], ccfg.initial_cash,
                                      ccfg.fee_rate, ccfg.min_fee)
    summ, etf_s = cfs.summarize(res.values, res.contributions), cfs.summarize(etf_v, etf_c)
    return {
        **summ, "Fees": res.fees_paid, "Trades": res.n_trades, "Blocked_Sells": res.blocked_sells,
        "Holdings_End": len(res.final_positions), "Delist_Events": res.delist_events,
        "Beats_Hurdle_Annualized": bool(summ["XIRR"] >= hurdle),
        "ETF0050_Lump_XIRR": etf_s["XIRR"], "ETF0050_Lump_MDD": etf_s["TWR_MDD"],
        "Beats_0050_Lump": bool(summ["Final_Value"] >= etf_s["Final_Value"]),
        "t_NW_vs_0050": s5.compute_excess_metrics(twr, cfs.weekly_twr(etf_v, etf_c), lags=4)["Excess_t_NW"],
        **{f"TUW_{k}": v for k, v in em.time_under_water((1.0 + twr).cumprod()).items()},
        **{f"Hurdle3y_{k}": v for k, v in em.rolling_hurdle_windows(twr, hurdle=hurdle).items()},
    }


def _log_trials(log: Path, cfg_hash: str, std: Dict[str, Dict[str, Any]], user: Dict[str, Dict[str, Any]]) -> int:
    for arm, m in std.items():
        psim.append_trial_log(log, stage="v8_FROZEN_HISTORICAL_OOS", market="TW", model=arm,
                              change=f"one-shot frozen v8 walk-forward, config {cfg_hash[:12]}",
                              ic=m["IC"], sharpe=m["Board_Sharpe"], mdd=m["MDD"], t_nw=m["Excess_t_NW"],
                              cagr=m["Board_CAGR"], turnover=m["Rebalance_Turnover"])
    for name, m in user.items():
        psim.append_trial_log(log, stage="v8_FROZEN_USER_RULES_100K", market="TW", model=name,
                              change=f"100k lump user rules, config {cfg_hash[:12]}; XIRR={m['XIRR']:.4f}",
                              ic=np.nan, sharpe=np.nan, mdd=m["TWR_MDD"], t_nw=m["t_NW_vs_0050"],
                              cagr=m["TWR_CAGR"])
    return int(pd.read_csv(log)["trial_id"].max())


def _market_series(dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    start = (dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    fx = rcs._next_day_value(rcs._yahoo_series("TWD=X", start, end, "Close"), dates)
    etf = rcs._next_day_value(rcs._yahoo_series("0050.TW", start, end, "Adj Close"), dates)
    return {"fx": fx, "etf_twd": etf, "etf_usd_w": _to_usd_returns(etf, fx)}


def _slices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {"full_oos": df, "pre_holdout": df.loc[df["date"] <= HOLDOUT_CUTOFF]}


def run(panel: pd.DataFrame, cfg: Dict[str, Any], price_wide: Optional[pd.DataFrame], trial_log: Path) -> Dict[str, Any]:
    require_frozen(cfg)
    scored = arm_scores(walkforward_ridge(panel, cfg).dropna(subset=["v8_ridge"]), cfg)
    scored = scored.loc[scored["market"] == "TW"].copy()
    dates = pd.DatetimeIndex(sorted(scored["date"].unique()))
    mk = _market_series(dates)
    ew_w = scored.groupby("date")["fwd_ret_1w_open"].mean().dropna()
    scored["px_twd"] = scored["next_open_usd"] * scored["date"].map(mk["fx"])
    fallback = None if price_wide is None else price_wide.reindex(dates).mul(mk["fx"], axis=0)
    hurdle = float(cfg["portfolios"]["user_rules"]["hurdle_annual"])
    user = {f"{arm}|max_names={n}": user_rules_eval(scored, arm, user_rules_config(cfg, n), fallback, mk["etf_twd"], hurdle)
            for arm, n in USER_VARIANTS}
    arms = [a for a in ARM_SCORES if a in cfg["arms"]]
    provisional = {a: standard_eval(scored, f"{a}_score", ew_w, mk["etf_usd_w"], 1) for a in arms}
    n_trials = _log_trials(trial_log, vc.config_hash(cfg), provisional, user) + EXTRA_TRIALS
    std = {sl: {a: standard_eval(sub, f"{a}_score", ew_w, mk["etf_usd_w"], n_trials) for a in arms}
           for sl, sub in _slices(scored).items()}
    primary = next(a for a, v in cfg["arms"].items() if v["primary"])
    return {"config_sha256": vc.config_hash(cfg), "code_commit": cfg["code_commit"], "n_trials_cons": n_trials,
            "oos_start": str(dates.min().date()), "oos_end": str(dates.max().date()),
            "gate_primary_full_oos": std["full_oos"][primary]["Pass_Gate"], "standard": std, "user_rules": user,
            "scored": scored}


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot historical OOS of the frozen Snowball v8 config")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--price-wide", type=Path, default=None)
    parser.add_argument("--trial-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = vc.load_config(args.config)
    require_frozen(cfg)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"{args.out_dir} is not empty: the historical OOS is one-shot")
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    wide = None
    if args.price_wide is not None:
        wide = pd.read_parquet(args.price_wide)
        wide.index = pd.to_datetime(wide.index).astype("datetime64[ns]")
    out = run(panel, cfg, wide, args.trial_log)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["date", "ticker", "industry_twse", "v8_zero", "v8_ridge", "pred_baseline_v6_prior", "model_train_end"]
    out.pop("scored")[keep + [c for c in out_cols(cfg)]].to_parquet(args.out_dir / "v8_oos_scores.parquet", index=False)
    (args.out_dir / "v8_oos_summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("standard", "user_rules")}, indent=1))


def out_cols(cfg: Dict[str, Any]) -> List[str]:
    return [f"{a}_score" for a in ARM_SCORES if a in cfg["arms"]]


if __name__ == "__main__":
    main()


__all__ = ["require_frozen", "walkforward_ridge", "arm_scores", "standard_eval", "user_rules_config", "user_rules_eval"]
