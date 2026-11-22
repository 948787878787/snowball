"""Snowball v8 trained arm: sign-constrained Ridge with frozen hyper-parameters (no tuning).

Design matrix (all cross-sectional z, NaN -> 0 = neutral):
  sa_<input>        every v8 dimension input multiplied by its pre-registered sign, coefficient >= 0
  <extra feature>   `trained.extra_features` {name: sign}; sign +1 -> >= 0, -1 -> <= 0, 0 -> free
  dim_<d>_x_stress  each dimension score x macro_stress (state conditioning), free
Targets: one Ridge per `trained.targets`; the arm's prediction is their average.
Training rows: labels dated <= train_end - label_embargo_weeks (labels must be complete).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import v8_composite as vc

Bound = Tuple[Optional[float], Optional[float]]
_BOUND_OF_SIGN: Dict[int, Bound] = {1: (0.0, None), -1: (None, 0.0), 0: (None, None)}


def design_matrix(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[Bound]]:
    zc = cfg["zscore"]
    cols: Dict[str, np.ndarray] = {}
    bounds: Dict[str, Bound] = {}

    def z(col: str) -> np.ndarray:
        return vc.cs_zscore(df, col, zc["winsor"], zc["min_names"], zc["by"], zc.get("winsor_pct", 0.01)).fillna(0.0).to_numpy(dtype=float)

    for inputs in cfg["dimensions"].values():
        for col, sign in inputs.items():
            cols[f"sa_{col}"] = sign * z(col)
            bounds[f"sa_{col}"] = (0.0, None)
    for col, sign in cfg["trained"].get("extra_features", {}).items():
        cols[col] = z(col)
        bounds[col] = _BOUND_OF_SIGN[int(sign)]
    stress = df["macro_stress"].fillna(0.0).clip(-3.0, 3.0).to_numpy(dtype=float)
    for name in cfg["dimensions"]:
        key = f"dim_{name}_x_stress"
        cols[key] = df[f"dim_{name}"].fillna(0.0).to_numpy(dtype=float) * stress
        bounds[key] = (None, None)
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names, [bounds[n] for n in names]


def _ridge_bounded(X: np.ndarray, y: np.ndarray, alpha: float, bounds: List[Bound]) -> Tuple[float, np.ndarray]:
    """min ||X b - (y - mean y)||^2 / n + alpha/n ||b||^2 subject to bounds (L-BFGS-B on the Gram)."""
    n = max(len(y), 1)
    y_mean = float(np.mean(y))
    gram = X.T @ X / n + (alpha / n) * np.eye(X.shape[1])
    rhs = X.T @ (y - y_mean) / n

    def obj(b: np.ndarray) -> Tuple[float, np.ndarray]:
        g = gram @ b
        return float(b @ g - 2.0 * rhs @ b), 2.0 * (g - rhs)

    res = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Ridge optimisation failed: {res.message}")
    return y_mean, res.x


def fit(df: pd.DataFrame, cfg: Dict[str, Any], train_end: pd.Timestamp) -> Dict[str, Any]:
    X, names, bounds = design_matrix(df, cfg)
    return fit_design(X, names, bounds, df, cfg, train_end)


def fit_design(X: np.ndarray, names: List[str], bounds: List[Bound], df: pd.DataFrame, cfg: Dict[str, Any],
               train_end: pd.Timestamp) -> Dict[str, Any]:
    """Fit on a precomputed design matrix (rows aligned with `df`). The cross-sectional z-scores are
    per date, so computing them once for the whole panel carries no look-ahead."""
    tc = cfg["trained"]
    last = pd.Timestamp(train_end) - pd.Timedelta(weeks=int(tc["label_embargo_weeks"]))
    in_window = (df["date"] <= last).to_numpy()
    model: Dict[str, Any] = {"features": names, "intercept": {}, "coef": {}, "n_rows": {},
                             "l2_alpha": float(tc["l2_alpha"])}
    for tgt in tc["targets"]:
        rows = in_window & df[tgt].notna().to_numpy()
        if rows.sum() < X.shape[1] * 10:
            raise ValueError(f"Too few training rows for {tgt}: {int(rows.sum())}")
        b0, b = _ridge_bounded(X[rows], df.loc[rows, tgt].to_numpy(dtype=float), float(tc["l2_alpha"]), bounds)
        model["intercept"][tgt], model["coef"][tgt], model["n_rows"][tgt] = b0, b.tolist(), int(rows.sum())
    used = df.loc[in_window & df[tc["targets"][0]].notna().to_numpy(), "date"]
    model["train_last_date"] = str(pd.Timestamp(used.max()).date())
    return model


def predict(df: pd.DataFrame, cfg: Dict[str, Any], model: Dict[str, Any]) -> np.ndarray:
    X, names, _ = design_matrix(df, cfg)
    if names != model["features"]:
        raise ValueError("Model feature list does not match the config design matrix")
    preds = [model["intercept"][t] + X @ np.asarray(model["coef"][t]) for t in cfg["trained"]["targets"]]
    return np.mean(preds, axis=0)


def save_model(model: Dict[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(model, indent=1), encoding="utf-8")


def load_model(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    """Fit the frozen Ridge arm on every label completed before --train-end and save it as JSON."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Fit the Snowball v8 frozen Ridge arm (no tuning)")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-end", default=None, help="Defaults to the latest panel date")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = vc.load_config(args.config)
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    end = pd.Timestamp(args.train_end) if args.train_end else panel["date"].max()
    model = {**fit(panel, cfg, end), "config_sha256": vc.config_hash(cfg)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out)
    logging.info("Saved %s rows=%s last_label_date=%s", args.out, model["n_rows"], model["train_last_date"])


if __name__ == "__main__":
    main()
