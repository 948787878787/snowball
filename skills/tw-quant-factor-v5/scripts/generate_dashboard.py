#!/usr/bin/env python3
"""
Generic Open-Source v5 HTML Dashboard Generator (Zero-PII).

Reads any local portfolio JSON (see references/portfolio-template.json), computes latest
v5 multi-factor rankings from backtest_v5, and writes a self-contained Dark/Light HTML
dashboard with localStorage persistence, editable holdings, an odd-lot order calculator,
and an interactive 10-year compounding simulator.

Usage:
    python generate_dashboard.py --portfolio ./my_portfolio.json --output ./v5_dashboard.html
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
      --background: #090d16;
      --card: #111827;
      --secondary: #1a2333;
      --border: #243044;
      --foreground: #f1f5f9;
      --muted-foreground: #94a3b8;
      --primary: #10b981;
      --primary-foreground: #022c22;
      --secondary-foreground: #e2e8f0;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --background: #f8fafc;
        --card: #ffffff;
        --secondary: #f1f5f9;
        --border: #e2e8f0;
        --foreground: #0f172a;
        --muted-foreground: #64748b;
        --primary: #0f172a;
        --primary-foreground: #ffffff;
        --secondary-foreground: #334155;
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


def build_payload(cfg: dict) -> dict:
    p = load_panels()
    v5 = build_v5_panels(p)
    close_df = data.load_prices()["close"]
    val_df = data.load_valuation().sort_values("date").groupby("stock_id").last()
    gmap = group_of()

    latest_daily_dt = close_df.index[-1].strftime("%Y-%m-%d")
    latest_panel_dt = v5["v5_income"].index[-1]
    s_inc = v5["v5_income"].loc[latest_panel_dt].dropna().sort_values(ascending=False)

    holdings_out = []
    for h in cfg.get("holdings", []):
        code = str(h["code"])
        px = (
            round(float(close_df[code].dropna().iloc[-1]), 2)
            if code in close_df.columns and not close_df[code].dropna().empty
            else float(h.get("fallback_price", h["cost"]))
        )
        holdings_out.append({
            "mkt": str(h.get("market", "TW")).upper(),
            "code": code,
            "name": h.get("name", code),
            "shares": float(h["shares"]),
            "cost": float(h["cost"]),
            "price": px,
            "targetSell": h.get("target_sell"),
            "sellList": h.get("target_sell") is not None,
            "action": h.get("note", "Core DCA" if h.get("role") == "core_dca" else "Monitor"),
        })

    watch_codes = [str(w["code"]) for w in cfg.get("watchlist", [])]
    screener_out = []
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

    top_sat = screener_out[0]["code"] if screener_out else "5880"
    core_code = str(cfg.get("core_etf_code", "0056"))
    core_px = float(cfg.get("core_etf_price", 40.0))
    for h in holdings_out:
        if h["code"] == core_code:
            core_px = h["price"]

    return {
        "storageKey": "tw_quant_v5_open_dashboard_v1",
        "title": "tw-quant-factor-v5 實證選股與現金流工作台",
        "priceDate": latest_daily_dt,
        "modelDate": latest_panel_dt.strftime("%Y-%m-%d"),
        "fxUsdTwd": float(cfg.get("currency_fx_usd_twd", 32.5)),
        "defaultMonthly": int(cfg.get("monthly_inflow_twd", 10000)),
        "defaultCoreRatio": int(cfg.get("core_ratio_pct", 60)),
        "coreSymbol": {"code": core_code, "name": cfg.get("core_etf_name", "Core ETF"), "price": core_px},
        "defaultSatelliteCode": top_sat,
        "card3": {
            "title": "v5 樣本外 Jensen Alpha",
            "val": "+14.33%",
            "sub": "NW6 t = 2.16 · 夏普 2.08",
        },
        "card4": {
            "title": "v5 榜首衛星候選",
            "val": top_sat,
            "sub": "點選「v5 選股與買單計算機」試算",
        },
        "watchCodes": watch_codes,
        "holdings": holdings_out,
        "screener": screener_out,
        "auditModels": [
            {"name": "v3 平滑版", "ic": 0.0773, "icT": 3.93, "longGross": 11.2, "beta": 1.049, "alpha": 10.36, "t": 1.52, "mdd": -18.1, "sharpe": 1.88, "turn": 31.9, "sectors": 10.1, "hl": False},
            {"name": "v4 高息版", "ic": 0.0924, "icT": 4.57, "longGross": 12.3, "beta": 0.952, "alpha": 10.31, "t": 1.46, "mdd": -15.7, "sharpe": 1.84, "turn": 38.1, "sectors": 9.1, "hl": False},
            {"name": "v5 純三因子", "ic": 0.0950, "icT": 5.14, "longGross": 14.0, "beta": 1.166, "alpha": 11.15, "t": 1.69, "mdd": -23.2, "sharpe": 1.88, "turn": 27.2, "sectors": 9.9, "hl": False},
            {"name": "v5 高息防禦版 ★", "ic": 0.0960, "icT": 5.63, "longGross": 14.6, "beta": 0.961, "alpha": 14.33, "t": 2.16, "mdd": -17.4, "sharpe": 2.08, "turn": 24.8, "sectors": 11.2, "hl": True},
        ],
        "simBench": [
            {"name": "v5 高息防禦 (只買不賣+贏家減碼)", "val": 204200, "roi": "+104.2%", "color": "bg-emerald-500"},
            {"name": "v5 純三因子 (只買不賣+贏家減碼)", "val": 198800, "roi": "+98.8%", "color": "bg-sky-500"},
            {"name": "基準1：84 檔等權定期定額", "val": 194200, "roi": "+94.2%", "color": "bg-slate-400"},
            {"name": "核心+衛星 60/40 (低集中度 ★)", "val": 187700, "roi": "+87.7%", "color": "bg-amber-500"},
            {"name": "基準2：高息籃子 (ETF 代理)", "val": 178900, "roi": "+78.9%", "color": "bg-slate-500"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v5 interactive HTML dashboard.")
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=SKILL_DIR / "references" / "portfolio-template.json",
        help="Path to local portfolio JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v5_dashboard.html"),
        help="Output HTML path.",
    )
    args = parser.parse_args()

    cfg = json.loads(args.portfolio.read_text(encoding="utf-8"))
    payload = build_payload(cfg)
    html = TEMPLATE_HTML.read_text(encoding="utf-8")
    out_html = inject_data(html, payload).replace("<style>", f"{STANDALONE_THEME_CSS}\n  <style>", 1)
    args.output.write_text(out_html, encoding="utf-8")
    print(f"Generated interactive v5 dashboard: {args.output}")


if __name__ == "__main__":
    main()
