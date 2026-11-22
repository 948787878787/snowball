"""TDD tests for the v8 weekly paper-trading recorder (append-only, config-stamped, no orders)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import paper_trade as pt
import v8_model as vm

CFG = {
    "status": "FROZEN",
    "dimensions": {"a": {"x": 1}, "b": {"y": -1}},
    "dimension_weights": {"a": 0.5, "b": 0.5},
    "zscore": {"by": "date", "winsor": 3.0, "min_names": 3},
    "trained": {"l2_alpha": 1.0, "targets": ["t4"], "label_embargo_weeks": 1, "extra_features": {}},
    "arms": {"P1_v8_blend": {"primary": True}, "P2_v6_prior": {"primary": False}, "P3_v8_zero": {"primary": False}},
}


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-09-04", periods=12, freq="W-FRI")
    n = 40
    df = pd.DataFrame({
        "date": np.repeat(dates, n), "ticker": np.tile([f"{2000 + i}.TW" for i in range(n)], len(dates)),
        "industry": np.tile(["a", "b"], len(dates) * n // 2),
        "x": rng.normal(size=len(dates) * n), "y": rng.normal(size=len(dates) * n),
        "macro_stress": 0.0, "pred_baseline_v6_prior": rng.normal(size=len(dates) * n),
    })
    df["t4"] = df.groupby("date")["x"].rank(pct=True)
    import v8_composite as vc
    return vc.add_v8_zero(df, CFG)


def _model(panel):
    return vm.fit(panel, CFG, train_end=panel["date"].max())


def test_snapshot_scores_every_member_for_each_arm():
    panel = _panel()
    snap = pt.score_snapshot(panel, CFG, _model(panel))
    last = panel["date"].max()
    assert (snap["date"] == last).all() and len(snap) == 40
    for arm in ("P1_v8_blend", "P2_v6_prior", "P3_v8_zero"):
        pct = snap[f"{arm}_pct"]
        assert pct.between(0, 1).all() and pct.notna().all()


def test_record_week_writes_manifest_with_hash_and_refuses_overwrite(tmp_path: Path):
    panel = _panel()
    rec = pt.record_week(panel, CFG, _model(panel), tmp_path, commit="abc123", dry_run=False)
    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert manifest.iloc[0]["config_sha256"] == rec["config_sha256"]
    assert manifest.iloc[0]["commit"] == "abc123"
    assert (tmp_path / rec["snapshot_file"]).exists()
    with pytest.raises(FileExistsError):
        pt.record_week(panel, CFG, _model(panel), tmp_path, commit="abc123", dry_run=False)


def test_draft_config_refused_unless_dry_run(tmp_path: Path):
    panel = _panel()
    draft = {**CFG, "status": "DRAFT"}
    with pytest.raises(PermissionError):
        pt.record_week(panel, draft, _model(panel), tmp_path, commit="x", dry_run=False)
    rec = pt.record_week(panel, draft, _model(panel), tmp_path, commit="x", dry_run=True)
    assert rec["dry_run"] is True
    assert not (tmp_path / "manifest.csv").exists()


def test_watchlist_report_flags_sell_zone_and_top_candidates():
    panel = _panel()
    snap = pt.score_snapshot(panel, CFG, _model(panel))
    report = pt.watchlist_report(snap, ["2000", "2039", "9999"], arm="P1_v8_blend", sell_pct=0.60)
    assert list(report["stock_id"]) == ["2000", "2039", "9999"]
    row = report.set_index("stock_id")
    assert row.loc["9999", "status"] == "not in universe"
    expected = "sell zone" if row.loc["2000", "pct"] < 0.60 else "hold zone"
    assert row.loc["2000", "status"] == expected


def test_snapshot_has_no_return_columns():
    panel = _panel().assign(fwd_ret_4w=1.0, target_rank_4w=0.5)
    snap = pt.score_snapshot(panel, CFG, _model(_panel()))
    assert not any(c.startswith(("fwd_", "target_")) for c in snap.columns)


def test_snapshot_demeans_within_configured_industry_column():
    panel = _panel()
    panel["industry_twse"] = np.tile(["c", "d", "e", "f"], len(panel) // 4)
    cfg = {**CFG, "industry_col": "industry_twse"}
    snap = pt.score_snapshot(panel, cfg, _model(panel))
    assert "industry_twse" in snap.columns
    means = snap.groupby("industry_twse")["P1_v8_blend_score"].mean()
    assert np.allclose(means, 0.0)


def test_failed_oos_outcome_forces_observation_only_status(tmp_path: Path):
    panel = _panel()
    outcome = {"historical_oos": "FAILED", "paper_status": "FAILED_OOS_OBSERVATION_ONLY"}
    rec = pt.record_week(panel, CFG, _model(panel), tmp_path, commit="abc", dry_run=False, outcome=outcome)
    assert rec["status"] == "FAILED_OOS_OBSERVATION_ONLY"
    assert pd.read_csv(tmp_path / "manifest.csv").iloc[0]["status"] == "FAILED_OOS_OBSERVATION_ONLY"


def test_default_outcome_file_is_read_next_to_config(tmp_path: Path):
    (tmp_path / "v8_outcome.json").write_text('{"historical_oos": "FAILED", "paper_status": "FAILED_OOS_OBSERVATION_ONLY"}')
    assert pt.load_outcome(tmp_path / "v8_frozen.json")["paper_status"] == "FAILED_OOS_OBSERVATION_ONLY"
    assert pt.load_outcome(tmp_path / "missing" / "v8_frozen.json") == {}
