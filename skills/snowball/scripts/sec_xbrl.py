"""Point-in-Time SEC EDGAR XBRL Company Facts Parser (Split-Aware & Multi-Tag Union).

Fixes Stage 1 / P0-6 defects:
1. Unions across all matching XBRL tags (ordered by priority) instead of picking one tag by row count,
   preventing legacy tag freezes when companies adopt ASC 606 or change taxonomy tags (e.g., AAPL).
2. Derives Q4 via exact TTM identity (`FY_prev + sum(Q_cur) - sum(Q_prev)`) when 10-K filers omit
   standalone Q4 tags.
3. Extracts `dei:EntityCommonStockSharesOutstanding` (fallback `us-gaap` diluted/common shares) and
   `ttm_net_income` so downstream panel merging can compute split-immune market-cap E/P against
   unadjusted nominal `raw_close_usd`.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

REV_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "InterestAndDividendIncomeOperating",
    "TotalRevenuesAndOtherIncome",
)
GP_TAGS = ("GrossProfit",)
OPINC_TAGS = ("OperatingIncomeLoss",)
OCF_TAGS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
NI_TAGS = ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic")
EQ_TAGS = ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
DILUTED_SHARES_TAGS = (
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "CommonStockSharesOutstanding",
)
DEI_SHARES_TAGS = ("EntityCommonStockSharesOutstanding",)

VALID_FORMS = frozenset({"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F"})


def _extract_xbrl_series(
    facts_ns: dict, tags: Sequence[str], unit_key: str = "USD", is_duration: bool = True,
) -> pd.DataFrame:
    """Union valid XBRL observations across all matching tags, keeping point-in-time first filing per period."""
    rows: List[tuple] = []
    for prio, tag in enumerate(tags):
        node = facts_ns.get(tag)
        if not isinstance(node, dict):
            continue
        units = node.get("units", {})
        items = units.get(unit_key) or (next((v for k, v in units.items() if "USD" in k), None) if unit_key == "USD" else None)
        for it in items or ():
            if str(it.get("form", "")).upper() not in VALID_FORMS:
                continue
            filed, end, val = it.get("filed"), it.get("end"), it.get("val")
            start = it.get("start") if is_duration else None
            if not filed or not end or val is None or (is_duration and not start):
                continue
            try:
                rows.append((end, filed, start, float(val), prio))
            except (TypeError, ValueError):
                continue

    cols = ["end", "filed", "val", "days", "is_annual"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=["end", "filed", "start", "val", "prio"])
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["end", "filed"])
    if is_duration:
        days = (df["end"] - pd.to_datetime(df["start"], errors="coerce")).dt.days
        df = df.loc[days.between(65, 115) | days.between(330, 400)].copy()
        if df.empty:
            return pd.DataFrame(columns=cols)
        df["days"] = days.loc[df.index].astype(int)
    else:
        df["days"] = 0
    df["is_annual"] = df["days"] >= 300
    df = df.sort_values(["filed", "prio"]).drop_duplicates(subset=["end", "is_annual"], keep="first")
    return df[cols].sort_values(["filed", "end"]).reset_index(drop=True)


def _latest_asof(src: pd.DataFrame, end_dt: pd.Timestamp, filed_dt: pd.Timestamp, annual: bool) -> float:
    if src.empty:
        return np.nan
    sub = src[(src["end"] == end_dt) & (src["filed"] <= filed_dt) & (src["is_annual"] == annual)]
    return float(sub["val"].iloc[-1]) if not sub.empty else np.nan


def _prior_year_val(src: pd.DataFrame, end_dt: pd.Timestamp, filed_dt: pd.Timestamp, annual: bool) -> float:
    if src.empty:
        return np.nan
    target_end = end_dt - pd.Timedelta(days=365)
    sub = src[
        (src["filed"] <= filed_dt)
        & (src["is_annual"] == annual)
        & ((src["end"] - target_end).abs() <= pd.Timedelta(days=25))
    ]
    return float(sub["val"].iloc[-1]) if not sub.empty else np.nan


def _latest_balance_with_date(src: pd.DataFrame, filed_dt: pd.Timestamp) -> tuple[float, object]:
    if src.empty:
        return np.nan, None
    sub = src[src["filed"] <= filed_dt]
    if sub.empty:
        return np.nan, None
    return float(sub["val"].iloc[-1]), sub["end"].iloc[-1].date().isoformat()


def _latest_balance(src: pd.DataFrame, filed_dt: pd.Timestamp) -> float:
    val, _ = _latest_balance_with_date(src, filed_dt)
    return val


def _ttm_flow_sum(src: pd.DataFrame, filed_dt: pd.Timestamp) -> float:
    """Compute exact Point-in-Time TTM flow, deriving Q4 from 10-K when standalone Q4 is omitted."""
    if src.empty:
        return np.nan
    sub = src[src["filed"] <= filed_dt]
    if sub.empty:
        return np.nan

    q_sub = sub[~sub["is_annual"]].sort_values("end").drop_duplicates("end", keep="last")
    if len(q_sub) >= 4:
        last4 = q_sub.iloc[-4:]
        if (last4["end"].iloc[-1] - last4["end"].iloc[0]).days <= 320:
            return float(last4["val"].sum())

    a_sub = sub[sub["is_annual"]].sort_values("end").drop_duplicates("end", keep="last")
    if a_sub.empty:
        return np.nan
    fy_end = a_sub["end"].iloc[-1]
    fy_val = float(a_sub["val"].iloc[-1])

    # Derive TTM = FY_prev + sum(Q_cur_after_FY) - sum(matching Q_prev_year)
    cur_qs = q_sub[q_sub["end"] > fy_end]
    if 1 <= len(cur_qs) <= 3:
        prev_vals = [_prior_year_val(q_sub, q_end, filed_dt, False) for q_end in cur_qs["end"]]
        if all(np.isfinite(v) for v in prev_vals):
            return float(fy_val + cur_qs["val"].sum() - sum(prev_vals))
    return fy_val


def _ttm_eps_sum(src: pd.DataFrame, shares_src: pd.DataFrame, filed_dt: pd.Timestamp) -> float:
    """Compute TTM EPS while normalizing intra-window stock splits using diluted share counts."""
    if src.empty:
        return np.nan
    sub = src[src["filed"] <= filed_dt]
    if sub.empty:
        return np.nan
    q_sub = sub[~sub["is_annual"]].sort_values("end").drop_duplicates("end", keep="last")
    if len(q_sub) >= 4:
        last4 = q_sub.iloc[-4:]
        if (last4["end"].iloc[-1] - last4["end"].iloc[0]).days <= 320:
            if not shares_src.empty:
                sh_sub = shares_src[shares_src["filed"] <= filed_dt].sort_values("end").drop_duplicates("end", keep="last")
                sh_map = dict(zip(sh_sub["end"], sh_sub["val"]))
                latest_sh = sh_map.get(last4["end"].iloc[-1], np.nan)
                if np.isfinite(latest_sh) and latest_sh > 0:
                    scaled = []
                    for _, qrow in last4.iterrows():
                        q_sh = sh_map.get(qrow["end"], np.nan)
                        val = float(qrow["val"])
                        if np.isfinite(q_sh) and q_sh > 0:
                            ratio = latest_sh / q_sh
                            if ratio >= 1.75 or ratio <= 0.57:
                                val = val / ratio
                        scaled.append(val)
                    return float(sum(scaled))
            return float(last4["val"].sum())
    return _ttm_flow_sum(src, filed_dt)


def _build_filing_record(ticker: str, row: pd.Series, series: dict) -> dict:
    end_dt, filed_dt, is_ann, rev_now = row["end"], row["filed"], bool(row["is_annual"]), float(row["val"])
    rev_prev = _prior_year_val(series["rev"], end_dt, filed_dt, is_ann)
    rev_yoy = float(np.clip(rev_now / rev_prev - 1.0, -0.80, 2.00)) if (np.isfinite(rev_prev) and rev_prev > 0) else np.nan

    gp_now, gp_prev = _latest_asof(series["gp"], end_dt, filed_dt, is_ann), _prior_year_val(series["gp"], end_dt, filed_dt, is_ann)
    gm_now = (gp_now / rev_now) if np.isfinite(gp_now) else np.nan
    gm_prev = (gp_prev / rev_prev) if (np.isfinite(gm_now) and np.isfinite(gp_prev) and np.isfinite(rev_prev) and rev_prev > 0) else np.nan
    gm_chg = float(np.clip(gm_now - gm_prev, -0.25, 0.25)) if (np.isfinite(gm_now) and np.isfinite(gm_prev)) else np.nan

    ocf_now, capex_now = _latest_asof(series["ocf"], end_dt, filed_dt, is_ann), _latest_asof(series["capex"], end_dt, filed_dt, is_ann)
    op_now = _latest_asof(series["op"], end_dt, filed_dt, is_ann)
    if np.isfinite(ocf_now):
        fcf_margin = float(np.clip((ocf_now - (abs(capex_now) if np.isfinite(capex_now) else 0.0)) / rev_now, -0.60, 0.60))
    elif np.isfinite(op_now):
        fcf_margin = float(np.clip(op_now / rev_now, -0.60, 0.60))
    else:
        fcf_margin = np.nan

    rule40_pct = float((rev_yoy + fcf_margin) * 100.0) if (np.isfinite(rev_yoy) and np.isfinite(fcf_margin)) else np.nan
    ttm_ni, eq_now = _ttm_flow_sum(series["ni"], filed_dt), _latest_balance(series["eq"], filed_dt)
    roe = float(np.clip(ttm_ni / eq_now, -0.80, 1.20)) if (np.isfinite(ttm_ni) and np.isfinite(eq_now) and eq_now > 0) else np.nan
    shares_out, shares_date = _latest_balance_with_date(series["dei_sh"], filed_dt)
    if not np.isfinite(shares_out) or shares_out <= 0:
        shares_out, shares_date = _latest_balance_with_date(series["sh_dur"], filed_dt)
    sh_ref = series["sh_dur"] if not series["sh_dur"].empty else series["dei_sh"]
    return {
        "ticker": ticker, "market": "US",
        "available_date": (filed_dt + pd.Timedelta(days=1)).date().isoformat(),
        "period_end": end_dt.date().isoformat(),
        "shares_date": shares_date or end_dt.date().isoformat(),
        "rev_yoy": rev_yoy, "fcf_margin": fcf_margin, "rule40_pct": rule40_pct,
        "gross_margin_chg": gm_chg, "roe": roe, "ttm_net_income": ttm_ni,
        "shares_out": shares_out, "ttm_eps": _ttm_eps_sum(series["eps"], sh_ref, filed_dt), "ep_yield": np.nan,
    }


def parse_sec_company_facts(ticker: str, payload: dict, min_year: int = 2018) -> pd.DataFrame:
    """Convert one SEC EDGAR `companyfacts` JSON into strictly Point-in-Time fundamental snapshots."""
    facts = payload.get("facts", {})
    gaap = facts.get("us-gaap") or facts.get("ifrs-full") or {}
    dei = facts.get("dei") or {}
    if not isinstance(gaap, dict) or not gaap:
        return pd.DataFrame()

    rev_df = _extract_xbrl_series(gaap, REV_TAGS, unit_key="USD", is_duration=True)
    if rev_df.empty:
        return pd.DataFrame()

    dei_sh = _extract_xbrl_series(dei, DEI_SHARES_TAGS, unit_key="shares", is_duration=False)
    if dei_sh.empty:
        dei_sh = _extract_xbrl_series(gaap, DILUTED_SHARES_TAGS, unit_key="shares", is_duration=False)

    series = {
        "rev": rev_df,
        "gp": _extract_xbrl_series(gaap, GP_TAGS, unit_key="USD", is_duration=True),
        "op": _extract_xbrl_series(gaap, OPINC_TAGS, unit_key="USD", is_duration=True),
        "ocf": _extract_xbrl_series(gaap, OCF_TAGS, unit_key="USD", is_duration=True),
        "capex": _extract_xbrl_series(gaap, CAPEX_TAGS, unit_key="USD", is_duration=True),
        "ni": _extract_xbrl_series(gaap, NI_TAGS, unit_key="USD", is_duration=True),
        "eq": _extract_xbrl_series(gaap, EQ_TAGS, unit_key="USD", is_duration=False),
        "eps": _extract_xbrl_series(gaap, EPS_TAGS, unit_key="USD/shares", is_duration=True),
        "sh_dur": _extract_xbrl_series(gaap, DILUTED_SHARES_TAGS, unit_key="shares", is_duration=True),
        "dei_sh": dei_sh,
    }
    cand = rev_df[(rev_df["filed"] >= pd.Timestamp(f"{min_year}-01-01")) & (rev_df["val"] > 0)]
    records = [_build_filing_record(ticker, row, series) for _, row in cand.iterrows()]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("available_date").drop_duplicates(subset=["ticker", "available_date"], keep="last")
