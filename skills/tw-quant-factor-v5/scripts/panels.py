"""Build weekly point-in-time panels for every testable pillar.

Testable with free historical data: Peer, FinStmt (monthly revenue proxy), Flow (institutional),
Macro (lagged macro transmission), Chain (US semis lead), reversal filter.
Not testable: News (no historical news archive) and the binary-event gate (no earnings calendar).
"""

import numpy as np
import pandas as pd

import features as ft
from universe import PB_GROUPS, PEER_GROUPS, group_of

BETA_WINDOW = 52
BETA_MIN = 26
TURNOVER_DAYS = 252
FLOW_VOLUME_DAYS = 60
REV_YOY_CLIP = (-0.8, 2.0)


def _weekly_index(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Week-ends (Fri) whose week is complete: drop the current partial week."""
    weeks = pd.Series(daily_index, index=daily_index).resample("W-FRI").last().dropna()
    last_day = daily_index.max()
    return pd.DatetimeIndex([w for w in weeks.index if w <= last_day])


def _us_weekly(daily: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """US close known at TW Friday close = previous US session. Shift by one day then sample."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    return shifted.resample("W-FRI").last().reindex(weeks)


def _asof_weekly(series_by_date: pd.Series, weeks: pd.DatetimeIndex) -> pd.Series:
    s = series_by_date.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(weeks)).ffill().reindex(weeks)


def revenue_yoy(revenue: pd.DataFrame, weeks: pd.DatetimeIndex) -> tuple:
    """Trailing-3-month revenue YoY and its 3-month acceleration, as of each week."""
    yoy_cols, acc_cols = {}, {}
    for sid, g in revenue.groupby("stock_id"):
        g = g.assign(period=g["revenue_year"] * 12 + g["revenue_month"]).sort_values("period")
        g = g.drop_duplicates("period", keep="last").set_index("period")
        full = g.reindex(range(g.index.min(), g.index.max() + 1))
        rev3 = full["revenue"].rolling(3, min_periods=3).sum()
        yoy = rev3 / rev3.shift(12) - 1.0
        acc = yoy - yoy.shift(3)
        avail = full["available"]
        ok = avail.notna()
        yoy_cols[sid] = _asof_weekly(pd.Series(yoy[ok].to_numpy(), index=avail[ok]), weeks)
        acc_cols[sid] = _asof_weekly(pd.Series(acc[ok].to_numpy(), index=avail[ok]), weeks)
    return pd.DataFrame(yoy_cols), pd.DataFrame(acc_cols)


def _pivot_weekly(df: pd.DataFrame, value: str, weeks: pd.DatetimeIndex, how: str = "last") -> pd.DataFrame:
    wide = df.pivot_table(index="date", columns="stock_id", values=value, aggfunc="sum")
    wide = wide.resample("W-FRI").sum(min_count=1) if how == "sum" else wide.resample("W-FRI").last()
    return wide.reindex(weeks)


def build_panels(prices: dict, revenue: pd.DataFrame, valuation: pd.DataFrame, inst: pd.DataFrame) -> dict:
    close_d, vol_d = prices["close"], prices["volume"]
    weeks = _weekly_index(close_d.dropna(how="all").index)
    groups = {g: tuple(m for m in mem if m in close_d.columns) for g, mem in PEER_GROUPS.items()}

    close_w = close_d.resample("W-FRI").last().reindex(weeks)
    ret = close_w.pct_change(fill_method=None)
    mkt = prices["index"].resample("W-FRI").last().reindex(weeks).pct_change(fill_method=None)

    beta = ft.rolling_beta(ret, mkt, BETA_WINDOW, BETA_MIN)
    alpha = ret - beta.shift(1).mul(mkt, axis=0)
    target_1w = alpha.shift(-1)
    target_4w = sum(alpha.shift(-k) for k in range(1, 5))
    resid_sd = alpha.shift(1).rolling(BETA_WINDOW, min_periods=BETA_MIN).std()
    zabn = (alpha / resid_sd).clip(-3, 3)

    turnover = (close_d * vol_d).rolling(TURNOVER_DAYS, min_periods=60).mean()
    turnover = turnover.resample("W-FRI").last().reindex(weeks)

    peer_avg = ft.peer_mean_excl(alpha, groups)
    rel = alpha - peer_avg
    zrel = ft.ts_zscore(rel, BETA_WINDOW, BETA_MIN)
    lead, follower = ft.leader_alpha(alpha, turnover, groups)
    zg = ft.ts_zscore(lead - alpha, BETA_WINDOW, BETA_MIN)

    yoy, acc = revenue_yoy(revenue, weeks)
    yoy, acc = yoy.reindex(columns=alpha.columns), acc.reindex(columns=alpha.columns)
    fund = ft.cs_zscore(yoy.clip(*REV_YOY_CLIP))
    finstmt = ft.cs_zscore(0.5 * fund + 0.5 * ft.cs_zscore(acc.clip(-1.0, 1.0)).fillna(0.0).where(fund.notna()))
    dfund = fund - ft.peer_mean_excl(fund, groups)
    s = ft.smooth_switch(dfund, tau=0.5)

    val = _valuation_term(valuation, weeks, alpha.columns, groups)
    peer_v2 = ft.peer_score_v2(zg, follower, zrel, s, val["v"])
    peer_v1 = _peer_v1(alpha, lead, peer_avg, groups, dfund, val)

    flow = _flow_term(inst, vol_d, weeks, alpha.columns)
    macro = _macro_term(prices["macro"], alpha, weeks)
    chain = _chain_term(prices["us"], alpha, weeks)

    return {
        "weeks": weeks, "alpha": alpha, "target_1w": target_1w, "target_4w": target_4w,
        "zabn": zabn, "zrel": zrel, "zg": zg, "follower": follower, "s": s, "v": val["v"],
        "peer_v2": peer_v2, "peer_v1": peer_v1, "finstmt": finstmt, "flow": flow,
        "macro": macro, "chain": chain, "groups": groups,
    }


def _valuation_term(valuation, weeks, columns, groups) -> dict:
    yld = _pivot_weekly(valuation, "dividend_yield", weeks).reindex(columns=columns)
    per = _pivot_weekly(valuation, "PER", weeks).reindex(columns=columns)
    pbr = _pivot_weekly(valuation, "PBR", weeks).reindex(columns=columns)
    gmap = group_of()
    mult = pd.DataFrame({c: (pbr[c] if gmap.get(c) in PB_GROUPS else per[c]) for c in columns})
    mult = mult.where(mult > 0)
    peer_mult = ft.peer_mean_excl(mult, groups)
    yld_diff = yld - ft.peer_mean_excl(yld, groups)
    rel_mult = (mult - peer_mult) / peer_mult
    v = ft.cs_zscore(yld_diff).fillna(0.0) - 0.5 * ft.cs_zscore(rel_mult).fillna(0.0)
    return {"v": v.where(yld.notna() | mult.notna()), "yld_diff": yld_diff, "rel_mult": rel_mult}


def _peer_v1(alpha, lead, peer_avg, groups, dfund, val) -> pd.DataFrame:
    """Draft formula: cross-sectional dispersion denominator, raw leader return, raw valuation sum."""
    disp = pd.DataFrame(np.nan, index=alpha.index, columns=alpha.columns)
    for members in groups.values():
        sd = alpha[list(members)].std(axis=1)
        for c in members:
            disp[c] = sd
    zdisp = ((alpha - peer_avg) / disp.replace(0.0, np.nan)).clip(-10, 10)
    val_raw = (val["yld_diff"] / 100.0).fillna(0.0) - 0.5 * val["rel_mult"].fillna(0.0)
    return ft.peer_score_v1(ft.cs_zscore(lead), zdisp, dfund, val_raw)


def _flow_term(inst, vol_d, weeks, columns) -> pd.DataFrame:
    foreign = inst[inst["name"].isin(["Foreign_Investor", "Foreign_Dealer_Self"])]
    trust = inst[inst["name"] == "Investment_Trust"]
    f_w = _pivot_weekly(foreign, "net", weeks, how="sum").reindex(columns=columns)
    t_w = _pivot_weekly(trust, "net", weeks, how="sum").reindex(columns=columns)
    adv = vol_d.rolling(FLOW_VOLUME_DAYS, min_periods=20).mean().resample("W-FRI").last().reindex(weeks)
    denom = (adv.reindex(columns=columns) * 5).replace(0.0, np.nan)
    f_int = ft.cs_zscore(f_w / denom)
    t_int = ft.cs_zscore(t_w / denom)
    return ft.cs_zscore(f_int.fillna(0.0) + t_int.fillna(0.0)).where(f_int.notna())


def _macro_term(macro_d, alpha, weeks) -> pd.DataFrame:
    m = _us_weekly(macro_d, weeks)
    shocks = pd.DataFrame({
        "rate": m["^TNX"].diff(),
        "usd": m["DX-Y.NYB"].pct_change(fill_method=None),
        "oil": m["CL=F"].pct_change(fill_method=None),
    })
    total = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
    for col in shocks:
        b = ft.rolling_beta(alpha, shocks[col], BETA_WINDOW, BETA_MIN)
        total = total + ft.cs_zscore(b.mul(shocks[col], axis=0)).fillna(0.0)
    return ft.cs_zscore(total).where(alpha.notna())


def _chain_term(us_d, alpha, weeks) -> pd.DataFrame:
    sox = _us_weekly(us_d, weeks)["^SOX"].pct_change(fill_method=None)
    b = ft.rolling_beta(alpha, sox, BETA_WINDOW, BETA_MIN)
    return ft.cs_zscore(b.mul(sox, axis=0))
