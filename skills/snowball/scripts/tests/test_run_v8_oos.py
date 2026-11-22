"""TDD tests for the one-shot v8 historical walk-forward (leak-free refits, frozen-config guard)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import run_v8_oos as r8
import v8_composite as vc

CFG = {
    "status": "FROZEN", "code_commit": "abc123", "industry_col": "industry_twse",
    "dimensions": {"a": {"x": 1}, "b": {"y": -1}},
    "dimension_weights": {"a": 0.5, "b": 0.5},
    "zscore": {"by": "date", "winsor": 3.0, "min_names": 3},
    "trained": {"l2_alpha": 1.0, "targets": ["target_rank_4w"], "label_embargo_weeks": 5,
                "refit_every_weeks": 8, "min_train_weeks": 10, "extra_features": {}},
    "arms": {"P1_v8_blend": {"primary": True}, "P2_v6_prior": {"primary": False}, "P3_v8_zero": {"primary": False}},
}


def _panel(n_dates: int = 40, n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-07", periods=n_dates, freq="W-FRI")
    df = pd.DataFrame({
        "date": np.repeat(dates, n), "ticker": np.tile([f"{1100 + i}.TW" for i in range(n)], n_dates),
        "industry_twse": np.tile(["c", "d", "e"], n_dates * n // 3),
        "x": rng.normal(size=n_dates * n), "y": rng.normal(size=n_dates * n),
        "macro_stress": 0.0, "pred_baseline_v6_prior": rng.normal(size=n_dates * n),
    })
    df["target_rank_4w"] = df.groupby("date")["x"].rank(pct=True)
    return vc.add_v8_zero(df, CFG)


def test_walkforward_scores_only_come_from_models_fit_before_the_signal():
    panel = _panel()
    out = r8.walkforward_ridge(panel, CFG)
    scored = out.dropna(subset=["v8_ridge"])
    assert not scored.empty
    assert (scored["model_train_end"] <= scored["date"]).all()
    assert ((scored["date"] - scored["model_train_end"]) < pd.Timedelta(weeks=8)).all()
    embargo = pd.Timedelta(weeks=5)
    assert (pd.to_datetime(scored["model_label_last"]) <= scored["model_train_end"] - embargo).all()


def test_first_refit_waits_for_min_train_weeks_of_settled_labels():
    panel = _panel()
    out = r8.walkforward_ridge(panel, CFG)
    dates = sorted(panel["date"].unique())
    first = out.dropna(subset=["v8_ridge"])["date"].min()
    assert first == dates[10 + 5]          # 10 settled weeks + 5-week embargo


def test_arm_scores_are_demeaned_within_detailed_industry():
    panel = r8.walkforward_ridge(_panel(), CFG).dropna(subset=["v8_ridge"])
    scored = r8.arm_scores(panel, CFG)
    for arm in ("P1_v8_blend", "P2_v6_prior", "P3_v8_zero"):
        means = scored.groupby(["date", "industry_twse"])[f"{arm}_score"].mean()
        assert np.allclose(means, 0.0)


def test_run_refuses_unless_config_is_frozen_with_a_commit():
    with pytest.raises(PermissionError):
        r8.require_frozen({**CFG, "status": "DRAFT"})
    with pytest.raises(PermissionError):
        r8.require_frozen({**CFG, "code_commit": None})
    r8.require_frozen(CFG)


def test_run_end_to_end_on_synthetic_panel_logs_trials(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    panel = _panel(n_dates=60, n=60)
    panel["market"] = "TW"
    panel["industry"] = "x"
    panel["fwd_ret_1w_open"] = rng.normal(0.002, 0.03, len(panel))
    panel["next_open_usd"] = 3.0 + rng.uniform(0, 1, len(panel))
    cfg = {**CFG,
           "portfolios": {"user_rules": {"initial_capital_twd": 100000, "buy_amount_twd": 10000, "n_buy_per_week": 1,
                                         "buy_pct": 0.98, "sell_pct": 0.6, "max_weight": 0.2,
                                         "zero_realized_loss": True, "industry_cap": 0.3, "odd_lot_slippage": 0.003,
                                         "hurdle_annual": 0.06}},
           "costs": {"fee_rate": 0.001425, "min_fee_twd": 1.0, "sell_tax": 0.003}}
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    def fake_market(ds):
        fx = pd.Series(30.0, index=ds)
        etf = pd.Series(100 * np.cumprod(1 + rng.normal(0.002, 0.02, len(ds))), index=ds)
        return {"fx": fx, "etf_twd": etf, "etf_usd_w": r8._to_usd_returns(etf, fx)}

    monkeypatch.setattr(r8, "_market_series", fake_market)
    log = tmp_path / "trial_log.csv"
    pd.DataFrame([{"trial_id": 65, "stage": "seed"}]).to_csv(log, index=False)
    out = r8.run(panel, cfg, None, log)
    assert out["n_trials_cons"] == 65 + 3 + len(r8.USER_VARIANTS) + 20
    assert set(out["standard"]) == {"full_oos", "pre_holdout"}
    assert "Excess_t_NW" in out["standard"]["full_oos"]["P1_v8_blend"]
    assert len(out["user_rules"]) == len(r8.USER_VARIANTS)
    assert dates.min() < pd.Timestamp(out["oos_start"])
