"""Weekly entry-point check for the core ETF account: 6 lines of Traditional Chinese, suggestions only.

Uses only the plan's own rules (build-in tranches, drawdown buffer release, semiannual 5-point band,
buy-only top-ups from idle cash). No moving-average or volatility timing: etf_risk_v1 tested both and
neither passed its pre-registered gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

import etf_config as ec
import etf_ledger as el
import etf_portfolio as ep
import etf_rules as er

IDLE_CASH_MIN_SHARE = 0.01
DISCIPLINE = ("下單紀律：避開開盤前 15 分鐘與收盤前 5 分鐘；先看即時折溢價，00646 溢價 > 0.5% 不追；"
              "一律用盤中限價單。")


def latest_price_twd(cfg: Dict[str, Any], prices: pd.DataFrame, fx: pd.Series, as_of: pd.Timestamp) -> Dict[str, float]:
    """Last valid TWD price per risk target on or before `as_of` (markets close on different days)."""
    rate = float(fx.loc[:as_of].dropna().iloc[-1]) if not fx.empty else 1.0
    return {s: float(prices[s].loc[:as_of].dropna().iloc[-1]) * (rate if ec.is_usd(s) else 1.0)
            for s in ec.risk_targets(cfg)}


def _state(cfg, led: Optional[pd.DataFrame], prices, fx, as_of) -> Dict[str, Any]:
    if led is None:
        capital = float(cfg["initial_capital_twd"])
        return {"start": as_of, "cash": capital, "nav": capital, "values": {}, "invested": {}}
    values = ep.holdings_value(led, prices, fx, as_of)
    cash = el.cash_balance(led, as_of)
    return {"start": pd.Timestamp(cfg["contract"].get("start_date") or led["date"].min()), "cash": cash,
            "nav": cash + sum(values.values()), "values": values, "invested": el.net_invested(led, as_of)}


def _sym(s: str) -> str:
    return s.split(".")[0]


def _build_line(cfg, st, as_of, px) -> tuple:
    months, k = int(cfg["build_in"]["months"]), er.months_between(st["start"], as_of)
    plan = er.build_in(cfg, st["start"], as_of, st["invested"], px)
    remaining = sum(max(0.0, float(cfg["initial_capital_twd"]) * w - st["invested"].get(s, 0.0))
                    for s, w in ec.risk_targets(cfg).items()) if k < months else 0.0
    if k >= months:
        return "建倉：已完成，之後只做再平衡與回撤加碼", remaining
    if not plan:
        return f"建倉第 {k + 1}/{months} 批：本月已買足", remaining
    items = "、".join(f"{_sym(p['symbol'])} 約 {p['amount_twd']:,.0f} 元（約 {er.format_shares(p['shares'])} 股）" for p in plan)
    return f"建倉第 {k + 1}/{months} 批：本月待買 {items}", remaining


def _drawdown_line(cfg, bench, as_of, st, building: bool) -> str:
    dd = er.drawdown_deploy(cfg, bench.loc[:as_of], st["cash"], st["nav"])
    triggers = sorted(cfg["drawdown_buffer"]["triggers"], reverse=True)
    head = f"回撤：{_sym(cfg['drawdown_buffer']['index'])} 距 756 日高點 {dd['drawdown']:.1%}"
    nxt = next((t for t in triggers if dd["drawdown"] > t), None)
    gap = f"，距下一門檻 {nxt:.0%} 還差 {(dd['drawdown'] - nxt) * 100:.1f} 個百分點" if nxt is not None else ""
    if not dd["triggered"]:
        return head + gap
    hit = triggers[dd["triggered"] - 1]
    if building:
        return f"{head}，已觸發 {hit:.0%}；建倉中，照建倉計畫買即可{gap}"
    split = "／".join(f"{_sym(s)} 約 {a:,.0f} 元" for s, a in dd["split"].items())
    return f"{head}，已觸發 {hit:.0%}：可釋放現金緩衝約 {dd['deploy_twd']:,.0f} 元（{split}）{gap}"


def _rebalance_line(cfg, st, as_of, weights) -> str:
    every, months = int(cfg["rebalance"]["every_months"]), int(cfg["build_in"]["months"])
    m = er.months_between(st["start"], as_of)
    dev = "、".join(f"{'現金' if s == ec.CASH else _sym(s)} {weights.get(s, 0.0) - w:+.1%}" for s, w in cfg["targets"].items())
    if m >= months and m > 0 and m % every == 0:
        trades = er.rebalance(cfg, st["start"], as_of, weights, st["nav"])["trades"]
        todo = "；".join(f"{'買' if t['amount_twd'] > 0 else '賣'} {_sym(t['symbol'])} 約 {abs(t['amount_twd']):,.0f} 元"
                        for t in trades) or "都在 ±5% 內，不調整"
        return f"再平衡：本月是半年檢視月，{todo}（偏離 {dev}）"
    n = max(months, m + 1)
    n = n + (-n % every)
    due = (st["start"] + pd.DateOffset(months=n)).strftime("%Y-%m")
    if m < months:
        return f"再平衡：建倉完成後才檢查，第一次半年檢視 {due}"
    return f"再平衡：下次半年檢視 {due}；目前偏離 {dev}"


def _idle_cash_line(cfg, st, px, remaining_build: float) -> str:
    nav = st["nav"]
    excess = st["cash"] - float(cfg["targets"][ec.CASH]) * nav - remaining_build
    if excess <= IDLE_CASH_MIN_SHARE * nav:
        return "閒置現金：無（不需動作）"
    risk = ec.risk_targets(cfg)
    gaps = {s: max(0.0, w * nav - st["values"].get(s, 0.0)) for s, w in risk.items()}
    total = sum(gaps.values()) or 1.0
    weights = gaps if sum(gaps.values()) > 0 else risk
    norm = sum(weights.values())
    alloc = {s: min(excess * weights[s] / norm, gaps[s] if sum(gaps.values()) > 0 else excess) for s in risk}
    items = "、".join(f"{_sym(s)} 約 {a:,.0f} 元（約 {er.format_shares(er.shares_for(s, a, px[s]))} 股）"
                      for s, a in alloc.items() if a > 0)
    return f"閒置現金約 {excess:,.0f} 元：只買不賣，補低權重的 {items}（省證交稅與手續費）" if total else ""


def entry_check(cfg: Dict[str, Any], led: Optional[pd.DataFrame], prices: pd.DataFrame, fx: pd.Series,
                bench: pd.Series, as_of: pd.Timestamp) -> List[str]:
    px = latest_price_twd(cfg, prices, fx, as_of)
    st = _state(cfg, led, prices, fx, as_of)
    weights = {**{s: st["values"].get(s, 0.0) / st["nav"] for s in ec.risk_targets(cfg)}, ec.CASH: st["cash"] / st["nav"]}
    build, remaining = _build_line(cfg, st, as_of, px)
    building = er.months_between(st["start"], as_of) < int(cfg["build_in"]["months"])
    return [
        f"【切入點檢查 {as_of:%Y-%m-%d}】只給建議，不下單",
        build,
        _drawdown_line(cfg, bench, as_of, st, building),
        _rebalance_line(cfg, st, as_of, weights),
        _idle_cash_line(cfg, st, px, remaining),
        DISCIPLINE,
    ]


__all__ = ["latest_price_twd", "entry_check"]
