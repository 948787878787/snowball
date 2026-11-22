"""One-click monthly report for the core ETF account (5-8 lines of Traditional Chinese for the investor).

The report is a calculation aid. It suggests, never trades; hurdle numbers are estimates until the
contract anniversary settlement.

Usage (ledger OUTSIDE the git repo):
  python etf_monthly_report.py --init-ledger ../../../../core_etf_ledger/ledger.csv
  python etf_monthly_report.py --config ../../../configs/core_etf.json \
      --ledger ../../../../core_etf_ledger/ledger.csv [--month 2026-12] [--refresh]
  python etf_monthly_report.py --config ../../../configs/core_etf.json \
      --ledger ../../../../core_etf_ledger/ledger.csv --check-entry      # weekly entry-point check
Prices are cached once per day in <ledger dir>/.price_cache (use --refresh to re-download).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import etf_config as ec
import etf_entry_check as ck
import etf_hurdle as eh
import etf_ledger as el
import etf_portfolio as ep
import etf_rules as er
from etf_ledger import net_invested
from etf_price_cache import cached_series, yahoo_downloader


def previous_month(today: pd.Timestamp) -> str:
    return (today.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")


def _month_bounds(month: str):
    start = pd.Timestamp(f"{month}-01")
    return start, start + pd.offsets.MonthEnd(0)


def _suggestions(cfg, led, start, as_of, nav, weights, cash, prices, fx, bench) -> List[str]:
    nxt = as_of + pd.Timedelta(days=1)
    px_twd = ck.latest_price_twd(cfg, prices, fx, as_of)
    tips = [f"{s['reason']}：買 {s['symbol'].split('.')[0]} 約 {s['amount_twd']:,.0f} 元（約 {er.format_shares(s['shares'])} 股）"
            for s in er.build_in(cfg, start, nxt, net_invested(led, as_of), px_twd)]
    reb = er.rebalance(cfg, start, nxt, weights, nav)
    tips += [f"半年再平衡：{'買' if t['amount_twd'] > 0 else '賣'} {t['symbol'].split('.')[0]} 約 {abs(t['amount_twd']):,.0f} 元（{t['reason']}）"
             for t in reb["trades"]]
    if reb["due"] and not reb["trades"]:
        tips.append("半年再平衡檢查：各權重都在 ±5% 內，不調整")
    building = er.months_between(start, nxt) < int(cfg["build_in"]["months"])
    dd = er.drawdown_deploy(cfg, bench.loc[:as_of], cash, nav)
    if dd["triggered"] and not building:  # during build-in the uninvested cash is already scheduled
        tips.append(f"大盤自高點 {dd['drawdown']:.1%}，可分批動用現金緩衝約 {dd['deploy_twd']:,.0f} 元")
    return tips


def gather(cfg: Dict[str, Any], led: pd.DataFrame, prices: pd.DataFrame, fx: pd.Series, bench_adj: pd.Series,
           month: str) -> Dict[str, Any]:
    m0, m1 = _month_bounds(month)
    as_of = min(m1, prices.index.max())
    first = led["date"].min()
    start = pd.Timestamp(cfg["contract"].get("start_date") or first)
    nav = ep.nav_series(led, prices, fx, as_of)
    flows = el.external_flows(led)
    idx = ep.twr_index(nav, flows)
    cash = el.cash_balance(led, as_of)
    hv = ep.holdings_value(led, prices, fx, as_of)
    total = float(nav.iloc[-1])
    weights = {**{s: hv.get(s, 0.0) / total for s in ec.risk_targets(cfg)}, ec.CASH: cash / total}
    y0 = pd.Timestamp(f"{m0.year}-01-01")
    split = eh.contract_split(nav, flows, start, as_of, float(cfg["contract"]["hurdle_annual"]),
                              float(cfg["contract"]["manager_share_above_hurdle"]))
    acts = led[(led["date"] >= m0) & (led["date"] <= m1)]
    return {
        "month": month, "as_of": as_of, "nav": total, "weights": weights,
        "ret": {k: ep.period_return(idx, a, as_of) for k, a in (("m", m0), ("ytd", max(y0, first)), ("cum", first))},
        "bench": {k: ep.benchmark_return(bench_adj, a, as_of) for k, a in (("m", m0), ("ytd", max(y0, first)), ("cum", first))},
        "risk": ep.risk_metrics(idx), "hurdle": split[-1], "contract_year": len(split), "actions": acts,
        "tips": _suggestions(cfg, led, start, as_of, total, weights, cash, prices, fx, bench_adj),
    }


def _actions_text(acts: pd.DataFrame) -> str:
    parts = []
    for _, r in acts.iterrows():
        sym = r["symbol"].split(".")[0]
        if r["type"] in ("buy", "sell"):
            parts.append(f"{'買' if r['type'] == 'buy' else '賣'} {sym} {int(r['shares'])} 股")
        elif r["type"] in ("dividend", "interest"):
            parts.append(f"{'配息' if r['type'] == 'dividend' else '利息'} {r['amount_twd']:,.0f} 元")
        elif r["type"] in ("deposit", "withdraw", "carry"):
            parts.append(f"{ {'deposit': '入金', 'withdraw': '出金', 'carry': '支付分潤'}[r['type']] } {r['amount_twd']:,.0f} 元")
    return "、".join(parts) if parts else "無交易"


def render(d: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    r, b, h = d["ret"], d["bench"], d["hurdle"]
    alloc = " / ".join(f"{'現金' if s == ec.CASH else s.split('.')[0]} {w:.0%}" for s, w in d["weights"].items())
    target = "/".join(f"{w:.0%}".rstrip("%") for w in cfg["targets"].values())
    tips = "；".join(d["tips"]) if d["tips"] else "依計畫持有，無需操作"
    return [
        f"【專戶月報 {d['month'].replace('-', '/')}】",
        f"總淨值：{d['nav']:,.0f} 元（含現金與配息）｜本月 {r['m']:+.1%}｜今年 {r['ytd']:+.1%}｜累計 {r['cum']:+.1%}",
        f"同期 0050（含息）：本月 {b['m']:+.1%}｜今年 {b['ytd']:+.1%}｜累計 {b['cum']:+.1%}",
        f"風險：最大回撤 {d['risk']['mdd']:.1%}｜目前水下 {d['risk']['current_weeks_under_water']} 週",
        f"配置：{alloc}（目標 {target}）",
        f"6% 門檻試算（合約第 {d['contract_year']} 年至今 {h['year_return']:+.1%}，年度結算才算數）："
        f"出資方 {h['investor_gain']:+,.0f} 元｜操作方 {h['carry']:,.0f} 元",
        f"本月操作：{_actions_text(d['actions'])}",
        f"下月提示：{tips}",
    ]


def _fetch(cfg: Dict[str, Any], led: Optional[pd.DataFrame], end: pd.Timestamp, cache_dir: Optional[Path] = None,
           downloader=yahoo_downloader, refresh: bool = False):
    held = set() if led is None else set(led.loc[led["type"].isin(["buy", "sell"]), "symbol"])
    syms = sorted(set(ec.risk_targets(cfg)) | held)
    first = pd.Timestamp.today() if led is None else led["date"].min()
    start = (first - pd.Timedelta(days=1200)).strftime("%Y-%m-%d")
    stop = (pd.Timestamp(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    def get(sym: str, field: str) -> pd.Series:
        return cached_series(sym, field, start, stop, cache_dir, downloader, refresh)

    prices = pd.DataFrame({s: get(s, "Close") for s in syms})
    fx = get("TWD=X", "Close") if any(ec.is_usd(s) for s in syms) else pd.Series(dtype=float)
    bench = get(cfg["benchmark"], "Adj Close")
    return prices, fx, bench


def _load_ledger_or_none(path: Path) -> Optional[pd.DataFrame]:
    try:
        return el.load_ledger(path)
    except ValueError as exc:
        if "is empty" in str(exc):
            return None
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Core ETF account monthly investor report (no orders)")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--month", default=None, help="YYYY-MM; defaults to the previous calendar month")
    parser.add_argument("--init-ledger", type=Path, default=None, help="Create an empty ledger CSV and exit")
    parser.add_argument("--check-entry", action="store_true", help="Weekly entry-point check (5-8 lines)")
    parser.add_argument("--refresh", action="store_true", help="Ignore today's price cache and re-download")
    args = parser.parse_args()
    if args.init_ledger:
        el.init_ledger(args.init_ledger)
        print(f"Created {args.init_ledger}")
        return
    if not (args.config and args.ledger):
        parser.error("--config and --ledger are required")
    cfg, cache = ec.load_config(args.config), args.ledger.parent / ".price_cache"
    if args.check_entry:
        led = _load_ledger_or_none(args.ledger)
        prices, fx, bench = _fetch(cfg, led, pd.Timestamp.today(), cache, refresh=args.refresh)
        print("\n".join(ck.entry_check(cfg, led, prices, fx, bench, prices.index.max())))
        return
    led = el.load_ledger(args.ledger)
    month = args.month or previous_month(pd.Timestamp.today())
    prices, fx, bench = _fetch(cfg, led, _month_bounds(month)[1], cache, refresh=args.refresh)
    print("\n".join(render(gather(cfg, led, prices, fx, bench, month), cfg)))


if __name__ == "__main__":
    main()
