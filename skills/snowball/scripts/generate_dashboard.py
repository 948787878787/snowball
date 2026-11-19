#!/usr/bin/env python3
"""
Generic Open-Source Snowball Liquid Glass HTML Dashboard Generator (Zero-PII).

Reads any local portfolio JSON (see references/portfolio-template.json), computes latest
v5 multi-factor rankings, and writes a minimalist Liquid Glass HTML dashboard featuring
7 selectable rainbow glass tints (red/orange/yellow/green/blue/indigo/violet), Light/Dark/System
themes, editable monthly deposit & annual return rate, next-week outlook & action guidance,
discipline tasks, editable holdings, and a 10-year year-by-year compounding calculator.

Usage:
    python generate_dashboard.py --portfolio ./my_portfolio.json --output ./snowball_dashboard.html
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
TEMPLATE_HTML = SKILL_DIR / "resources" / "dashboard_template.html"
sys.path.insert(0, str(SCRIPT_DIR))

import data
from backtest_v3 import load_panels
from backtest_v5 import build_v5_panels
from universe import group_of

STANDALONE_THEME_CSS = """
  <style>
    :root {
      --background: #080c14;
      --card: #111827;
      --secondary: #1e293b;
      --border: rgba(255, 255, 255, 0.12);
      --foreground: #f8fafc;
      --muted-foreground: #94a3b8;
      --primary: #f8fafc;
      --primary-foreground: #0f172a;
      --glass-bg: rgba(17, 24, 39, 0.54);
      --glass-sub: rgba(30, 41, 59, 0.44);
      --glass-border: rgba(255, 255, 255, 0.13);
      --glass-highlight: rgba(255, 255, 255, 0.08);
      --orb-opacity: 0.26;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --background: #f5f7fb;
        --card: #ffffff;
        --secondary: #f1f5f9;
        --border: rgba(15, 23, 42, 0.10);
        --foreground: #0f172a;
        --muted-foreground: #64748b;
        --primary: #0f172a;
        --primary-foreground: #ffffff;
        --glass-bg: rgba(255, 255, 255, 0.62);
        --glass-sub: rgba(255, 255, 255, 0.46);
        --glass-border: rgba(255, 255, 255, 0.72);
        --glass-highlight: rgba(255, 255, 255, 0.85);
        --orb-opacity: 0.28;
      }
    }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC", "Noto Sans TC", sans-serif; }
  </style>
"""


def inject_data(html: str, payload: dict) -> str:
    """Replace the DASHBOARD_DATA JSON block between marker comments."""
    data_js = (
        "/*__DATA_START__*/\n    const DASHBOARD_DATA = "
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + ";\n    /*__DATA_END__*/"
    )
    return re.sub(r"/\*__DATA_START__\*/.*?/\*__DATA_END__\*/", lambda _: data_js, html, flags=re.S)


def _derive_tag_and_action(h: dict, price: float) -> tuple[bool, str, str]:
    target = h.get("target_sell")
    role = str(h.get("role", "hold")).lower()
    note = str(h.get("note", "")).strip()
    if target is not None:
        return True, "待賣", note or f"反彈至目標價 {target} 獲利減碼"
    if role == "core_dca":
        return False, "加碼", note or "核心標的：每月固定定期定額買進"
    if role == "frozen":
        return False, "凍結", note or "不賣也不加碼，靜待回穩"
    return False, "續抱", note or ("帳面正報酬，原地續抱觀察" if price >= float(h["cost"]) else "原地續抱領息觀察")


def build_payload(cfg: dict, offline: bool = False) -> dict:
    close_df = None
    val_df = None
    s_inc = None
    latest_daily_dt = "demo"
    gmap = group_of()

    if not offline:
        try:
            p = load_panels()
            v5 = build_v5_panels(p)
            close_df = data.load_prices()["close"]
            val_df = data.load_valuation().sort_values("date").groupby("stock_id").last()
            latest_daily_dt = close_df.index[-1].strftime("%Y-%m-%d")
            latest_panel_dt = v5["v5_income"].index[-1]
            s_inc = v5["v5_income"].loc[latest_panel_dt].dropna().sort_values(ascending=False)
        except Exception as exc:
            print(f"Notice: Live cache unavailable ({exc}); generating dashboard with portfolio fallback data.")

    holdings_out = []
    for h in cfg.get("holdings", []):
        code = str(h["code"])
        if close_df is not None and code in close_df.columns and not close_df[code].dropna().empty:
            px = round(float(close_df[code].dropna().iloc[-1]), 2)
        else:
            px = float(h.get("fallback_price", h["cost"]))
        sell_list, tag, action = _derive_tag_and_action(h, px)
        mkt = str(h.get("market", "TW")).upper()
        default_curr = {"US": "USD", "JP": "JPY", "EU": "EUR", "HK": "HKD", "UK": "GBP"}.get(mkt, "TWD")
        holdings_out.append({
            "mkt": mkt,
            "currency": str(h.get("currency", default_curr)).upper(),
            "code": code,
            "name": h.get("name", code),
            "shares": float(h["shares"]),
            "cost": float(h["cost"]),
            "price": px,
            "targetSell": h.get("target_sell"),
            "sellList": sell_list,
            "tag": h.get("tag", tag),
            "action": action,
        })

    watch_codes = [str(w["code"]) for w in cfg.get("watchlist", [])]
    screener_out = []
    if s_inc is not None and close_df is not None and val_df is not None:
        for rk, (sid, sc) in enumerate(s_inc.items(), 1):
            if rk <= 25 or sid in watch_codes:
                px = round(float(close_df[sid].dropna().iloc[-1]), 2)
                dy = round(float(val_df.loc[sid, "dividend_yield"]), 2) if sid in val_df.index else 0.0
                per = round(float(val_df.loc[sid, "PER"]), 2) if sid in val_df.index else 0.0
                screener_out.append({
                    "rank": rk,
                    "code": sid,
                    "name": sid,
                    "group": gmap.get(sid, ""),
                    "price": px,
                    "yield": dy,
                    "per": per,
                    "v5": round(float(sc), 3),
                })
    else:
        for idx, w in enumerate(cfg.get("watchlist", []), 1):
            screener_out.append({
                "rank": idx,
                "code": str(w["code"]),
                "name": w.get("name", str(w["code"])),
                "group": gmap.get(str(w["code"]), "watchlist"),
                "price": float(w.get("fallback_price", 50.0)),
                "yield": float(w.get("fallback_yield", 4.5)),
                "per": float(w.get("fallback_per", 12.0)),
                "v5": round(1.5 - idx * 0.2, 3),
            })

    base_currency = str(cfg.get("base_currency", "TWD")).upper()
    fx_to_base = cfg.get("fx_rates_to_base")
    if not isinstance(fx_to_base, dict):
        fx_to_base = (
            {"USD": 1.0, "TWD": 0.0308, "EUR": 1.08, "JPY": 0.0065, "HKD": 0.128, "GBP": 1.27}
            if base_currency == "USD"
            else {"TWD": 1.0, "USD": float(cfg.get("currency_fx_usd_twd", 32.5)), "EUR": 35.1, "JPY": 0.21, "HKD": 4.16, "GBP": 41.2}
        )

    default_monthly = int(cfg.get("monthly_inflow", cfg.get("monthly_inflow_twd", 10000)))
    core_ratio = float(cfg.get("core_ratio_pct", 60)) / 100.0
    top_sat = screener_out[0]["code"] if screener_out else "VT"

    weekly_outlook = cfg.get("weekly_outlook") or {
        "period": f"未來一週觀察（基準日 {latest_daily_dt}）",
        "posture": "紀律分批 · 逢回加碼不追高",
        "summary": (
            f"維持核心定投與只買不賣現金流再平衡；目前衛星觀察榜首為 {top_sat}。"
            "未來一週若市場震盪回測，優先以每月新現金流分批補齊低估值與低權重標的。"
        ),
        "actions": [
            {
                "label": "核心定投",
                "title": "按表執行每月核心買進",
                "detail": f"將每月 {int(round(core_ratio * 100))}% 資金固定投入核心 ETF，平滑波動並累積複利底倉。",
            },
            {
                "label": "衛星狙擊",
                "title": f"優先追蹤 {top_sat} 與前 20% 強勢股",
                "detail": "剩餘衛星資金優先流向 v5 排名靠前且估值合理之標的，分批逢低切入。",
            },
            {
                "label": "風控紀律",
                "title": "達標贏家減碼 · 虧損不亂砍",
                "detail": "僅在單一獲利持股觸及目標賣價或權重超標時減碼換股，絕不在低檔認賠。",
            },
        ],
    }

    tasks = cfg.get("tasks") or [
        {
            "id": "t1",
            "title": "按表執行本月核心 ETF 定期定額扣款",
            "desc": "固定投入核心比例資金，維持長期紀律與低交易成本。",
        },
        {
            "id": "t2",
            "title": f"追蹤衛星候選 {top_sat} 與觀察名單進場區間",
            "desc": "搭配 global_ranker.py 定期更新全球多因子排名。",
        },
        {
            "id": "t3",
            "title": "檢查待賣清單是否觸及目標獲利價",
            "desc": "遵守只買不賣現金流規則，獲利達標再換股轉入核心池。",
        },
    ]

    return {
        "storageKey": "snowball_glass_open_v2",
        "title": cfg.get("title", "Snowball 全球投資儀表板"),
        "subtitle": cfg.get("subtitle", "v5 實證多因子 · 只買不賣現金流再平衡"),
        "priceDate": latest_daily_dt,
        "baseCurrency": base_currency,
        "fxUsdTwd": float(cfg.get("currency_fx_usd_twd", 32.5)),
        "fxToBase": fx_to_base,
        "defaultMonthly": default_monthly,
        "defaultRate": float(cfg.get("expected_annual_return_pct", 8.5)),
        "defaultTint": str(cfg.get("default_tint", "blue")).lower(),
        "coreRatio": core_ratio,
        "coreLabel": cfg.get("core_label", "核心 ETF"),
        "tasksTitle": cfg.get("tasks_title", "每週紀律任務清單"),
        "weeklyOutlook": weekly_outlook,
        "tasks": tasks,
        "holdings": holdings_out,
        "screener": screener_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Snowball Liquid Glass HTML dashboard.")
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=SKILL_DIR / "references" / "portfolio-template.json",
        help="Path to local portfolio JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("snowball_dashboard.html"),
        help="Output HTML path.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Generate dashboard offline using fallback values without fetching market data.",
    )
    args = parser.parse_args()

    cfg = json.loads(args.portfolio.read_text(encoding="utf-8"))
    payload = build_payload(cfg, offline=args.offline)
    html = TEMPLATE_HTML.read_text(encoding="utf-8")
    out_html = inject_data(html, payload).replace("<style>", f"{STANDALONE_THEME_CSS}\n  <style>", 1)
    args.output.write_text(out_html, encoding="utf-8")
    print(f"Generated Snowball Liquid Glass dashboard: {args.output}")


if __name__ == "__main__":
    main()
