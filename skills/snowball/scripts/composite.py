"""Composite score S, reversal filter, walk-forward weights and probability calibration."""

import numpy as np
import pandas as pd

import evaluate as ev
import features as ft

PRIOR_WEIGHTS = {"macro": 0.20, "finstmt": 0.25, "peer": 0.15, "chain": 0.12, "flow": 0.15}
REV_LAMBDA_PRIOR = 0.3
REV_LAMBDA_GRID = (0.0, 0.15, 0.3, 0.5, 0.8)
PRIOR_KAPPA = float(np.log(0.55 / 0.45))
SHRINK = 0.5


def orthogonalize(y: pd.DataFrame, xs: list) -> pd.DataFrame:
    """Per-date cross-sectional OLS residual of y on xs (with intercept), re-z-scored."""
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for dt in y.index:
        yv = y.loc[dt]
        xv = pd.concat([x.loc[dt] for x in xs], axis=1)
        m = yv.notna() & xv.notna().all(axis=1)
        if m.sum() < 10:
            out.loc[dt] = yv
            continue
        a = np.column_stack([np.ones(m.sum()), xv[m].to_numpy()])
        coef, *_ = np.linalg.lstsq(a, yv[m].to_numpy(), rcond=None)
        resid = yv.copy()
        resid[m] = yv[m].to_numpy() - a @ coef
        out.loc[dt] = resid
    return ft.cs_zscore(out)


def pillar_panels(p: dict) -> dict:
    """Standardized, sequentially orthogonalized pillars: macro -> chain -> peer."""
    macro = p["macro"]
    chain = orthogonalize(p["chain"], [macro.fillna(0.0)])
    peer = orthogonalize(ft.cs_zscore(p["peer_v2"]), [macro.fillna(0.0), chain.fillna(0.0)])
    return {"macro": macro, "finstmt": p["finstmt"], "peer": peer, "chain": chain, "flow": p["flow"]}


def combine(pillars: dict, weights: dict, required: pd.DataFrame) -> pd.DataFrame:
    total = pd.DataFrame(0.0, index=required.index, columns=required.columns)
    for k, w in weights.items():
        total = total + w * pillars[k].reindex_like(required).fillna(0.0)
    return total.where(required.notna())


def ic_weights(prior: dict, icir: dict) -> dict:
    """w = 0.5*prior + 0.5*normalized max(ICIR,0). Falls back to prior if no positive ICIR."""
    pos = {k: max(icir.get(k, 0.0), 0.0) for k in prior}
    tot = sum(pos.values())
    p_tot = sum(prior.values())
    base = {k: v / p_tot for k, v in prior.items()}
    if tot <= 0:
        return base
    return {k: (1 - SHRINK) * base[k] + SHRINK * pos[k] / tot for k in prior}


def _icir(ic_all: pd.Series, dates) -> float:
    ic = ic_all[ic_all.index.isin(dates)]
    if len(ic) < 26 or ic.std() == 0:
        return 0.0
    return float(ic.mean() / ic.std())


def walk_forward(p: dict, pillars: dict, target: pd.DataFrame, oos_start, horizon: int, step: int = 4):
    """Refit weights / lambda / logistic every `step` weeks using only settled past data.

    Returns (score panel, prob panel, weight log).
    """
    weeks = target.index
    oos_weeks = weeks[weeks >= oos_start]
    first_valid = p["peer_v2"].dropna(how="all").index.min()
    score = pd.DataFrame(np.nan, index=weeks, columns=target.columns)
    prob = score.copy()
    log = []
    ic_by_pillar = {k: ev.rank_ic(pillars[k], target, min_names=20) for k in PRIOR_WEIGHTS}
    for i in range(0, len(oos_weeks), step):
        block = oos_weeks[i:i + step]
        refit_at = block[0]
        settled = weeks[(weeks >= first_valid) & (weeks <= refit_at - pd.Timedelta(weeks=horizon))]
        icir = {k: _icir(ic_by_pillar[k], settled) for k in PRIOR_WEIGHTS}
        w = ic_weights(PRIOR_WEIGHTS, icir)
        raw = combine(pillars, w, p["peer_v2"])
        lam = _best_lambda(raw, p["zabn"], target, settled)
        s_all = ft.reversal_filter(raw, p["zabn"], lam=lam)
        a, k = _fit_prob(s_all, target, settled)
        score.loc[block] = s_all.loc[block]
        prob.loc[block] = ft.sigmoid(a + k * s_all.loc[block])
        log.append({"refit": refit_at.date().isoformat(), **{f"w_{k}": round(v, 3) for k, v in w.items()},
                    "lambda": lam, "a": round(a, 4), "kappa": round(k, 4)})
    return score, prob, pd.DataFrame(log)


def _best_lambda(raw, zabn, target, dates) -> float:
    best, best_ic = REV_LAMBDA_PRIOR, -np.inf
    for lam in REV_LAMBDA_GRID:
        s = ft.reversal_filter(raw.loc[dates], zabn.loc[dates], lam=lam)
        ic = ev.rank_ic(s, target.loc[dates], min_names=20).mean()
        if ic > best_ic:
            best, best_ic = lam, ic
    return best


def _stack(signal: pd.DataFrame, target: pd.DataFrame, dates):
    s = signal.loc[dates].stack()
    t = target.loc[dates].stack()
    j = pd.concat([s, t], axis=1, keys=["s", "t"]).dropna()
    return j["s"].to_numpy(), (j["t"] > 0).astype(float).to_numpy()


def _fit_prob(signal, target, dates):
    x, y = _stack(signal, target, dates)
    if len(x) < 500:
        return 0.0, PRIOR_KAPPA
    return ev.fit_logistic(x, y)


def prior_model(p: dict, pillars: dict) -> tuple:
    """Doc v2 spec as written: prior weights, lambda 0.3, kappa=ln(.55/.45), p clipped to [0.40, 0.62]."""
    raw = combine(pillars, {k: v / sum(PRIOR_WEIGHTS.values()) for k, v in PRIOR_WEIGHTS.items()}, p["peer_v2"])
    s = ft.reversal_filter(raw, p["zabn"], lam=REV_LAMBDA_PRIOR)
    return s, ft.sigmoid(PRIOR_KAPPA * s).clip(0.40, 0.62)
