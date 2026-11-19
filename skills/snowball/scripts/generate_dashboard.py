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
      --glass-pop: rgba(17, 24, 39, 0.88);
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
        --glass-pop: rgba(255, 255, 255, 0.90);
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
        return True, "Sell Watch", note or f"Trim profit near target {target}"
    if role == "core_dca":
        return False, "Buy DCA", note or "Core holding — regular monthly DCA"
    if role == "frozen":
        return False, "Freeze", note or "No new adds — hold steady"
    return False, "Hold", note or ("In profit — hold steady" if price >= float(h["cost"]) else "Hold and monitor")


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

    default_currency = str(cfg.get("default_currency", cfg.get("base_currency", "USD"))).upper()
    fx_per_usd = cfg.get("fx_per_usd") or {
        "USD": 1.0,
        "TWD": float(cfg.get("currency_fx_usd_twd", 32.5)),
        "EUR": 0.925,
        "JPY": 154.0,
        "HKD": 7.8,
        "KRW": 1390.0,
        "GBP": 0.785,
    }

    default_monthly = int(cfg.get("monthly_inflow", cfg.get("monthly_inflow_twd", 1000)))
    default_monthly_cur = "USD" if "monthly_inflow" in cfg else ("TWD" if "monthly_inflow_twd" in cfg else default_currency)
    core_ratio = float(cfg.get("core_ratio_pct", 60)) / 100.0
    top_sat = screener_out[0]["code"] if screener_out else "VT"

    weekly_outlook = cfg.get("weekly_outlook") or {
        "period": f"Next 7 Days (Close {latest_daily_dt})",
        "posture": "Disciplined DCA · Buy Dips Only",
        "summary": (
            f"Keep deploying scheduled cash flow into core ETFs while monitoring top v5 satellite candidate {top_sat}. "
            "Use pullbacks to build underweighted top-20% positions and avoid chasing short-term rallies."
        ),
        "actions": [
            {
                "label": "Core DCA",
                "title": "Execute scheduled core buy",
                "detail": f"Deploy {int(round(core_ratio * 100))}% of monthly inflow into core ETFs to smooth volatility.",
            },
            {
                "label": "Satellite",
                "title": f"Track {top_sat} & top-20% leaders",
                "detail": "Direct remaining satellite funds to high-scoring, attractively valued global leaders on dips.",
            },
            {
                "label": "Risk Rule",
                "title": "Trim winners at target only",
                "detail": "Never panic sell drawdowns; rotate profits only when targets or weight caps trigger.",
            },
        ],
    }

    tasks = cfg.get("tasks") or [
        {
            "id": "t1",
            "title": "Complete monthly Core ETF DCA order",
            "desc": "Deploy core allocation on schedule to maintain low cost and compounding discipline.",
        },
        {
            "id": "t2",
            "title": f"Review satellite candidate {top_sat} & global v5 rankings",
            "desc": "Run global_ranker.py across US, EU, JP, HK, and TW universes.",
        },
        {
            "id": "t3",
            "title": "Check profit-target triggers on sell-watch list",
            "desc": "Follow buy-only cash-flow rebalancing and rotate realized gains into core holdings.",
        },
    ]

    return {
        "storageKey": "snowball_glass_open_v4",
        "defaultLang": str(cfg.get("default_lang", "en")).lower(),
        "defaultCurrency": default_currency,
        "defaultTint": str(cfg.get("default_tint", "blue")).lower(),
        "priceDate": latest_daily_dt,
        "fxPerUsd": fx_per_usd,
        "defaultMonthly": default_monthly,
        "defaultMonthlyCurrency": default_monthly_cur,
        "defaultRate": float(cfg.get("expected_annual_return_pct", 8.5)),
        "coreRatio": core_ratio,
        "coreLabel": cfg.get("core_label", "ETF"),
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
