---
name: snowball
description: >-
  All-agents global equity multi-factor ranking (v5) and buy-only cash-flow rebalancing skill.
  Covers Global markets (US, Europe, Japan, Hong Kong, Taiwan, or arbitrary custom tickers via Yahoo Finance).
  Use when the user asks to rank global or regional equities using empirical factors (Residual Momentum,
  Full-Group Industry Momentum, Revenue Growth, and 8%-Capped Industry-Neutral Value), run point-in-time
  4-cohort staggered backtests with weekly mark-to-market MDD, score probabilistic forecasts (Brier Score / BSS),
  or generate a minimalist Liquid Glass interactive HTML portfolio dashboard (with 7 rainbow glass tints:
  red, orange, yellow, green, blue, indigo, violet) and buy-only cash-flow rebalancing rules.
---

# Snowball — Global Equity Multi-Factor & Cash-Flow Rebalancing Skill (v5)

An agent-agnostic quantitative skill for **global equities** (`US`, `EU`, `JP`, `HK`, `TW`, `GLOBAL`, or custom tickers), compatible with **Antigravity (AGY)**, **Claude Code**, **OpenAI Codex CLI**, **Cursor**, **Windsurf**, **Cline**, and **OpenCode**.

> **Disclaimer**: Strictly for quantitative research, backtesting methodology, and systematic tracking. Not licensed financial advice.

## Supported Global Markets (`global_universe.py` & `resources/universes/*.json`)

Built-in database covers **2,780+ listed equities across 5 regional markets × 13 standardized industries (≥ 20 companies/industry; 1,000 US + 1,000 TW)** and is infinitely extensible via `manage_universe.py` or custom JSON files:

| Market Flag | Coverage | Stocks / Industries | Benchmark | Round-Trip Cost Model |
|---|---|---|---|---|
| `--market US` | NYSE / NASDAQ top 1,000 companies across 13 sectors | **1,000** (13 × ≥30) | `SPY` | `0.10%` |
| `--market TW` | TWSE / TPEx top 1,000 companies (`.TW` / `.TWO`) | **1,000** (13 × ≥20) | `^TWII` | `0.585%` (+ FinMind monthly rev in `backtest_v5.py`) |
| `--market EU` | Euronext / XETRA / LSE / SIX / Nordic | **260** (13 × 20) | `^STOXX50E` | `0.30%` |
| `--market JP` | Tokyo Stock Exchange (`.T`) | **260** (13 × 20) | `^N225` | `0.20%` |
| `--market HK` | Hong Kong Exchange (`.HK`) | **260** (13 × 20) | `^HSI` | `0.45%` (incl. stamp duty) |
| `--market GLOBAL` | Auto-aggregated across all regional markets | **2,780+** (13 × ≥115) | `ACWI` | `0.30%` |
| `--tickers ...` / `--universe-file` | Any custom Yahoo Finance symbols or JSON | Unlimited | Custom (`ACWI`) | `0.20%` |

Expand the database anytime with `manage_universe.py`:
```bash
.venv/bin/python skills/snowball/scripts/manage_universe.py --list
.venv/bin/python skills/snowball/scripts/manage_universe.py --market US --industry software_internet --add APP,MDB,NET
.venv/bin/python skills/snowball/scripts/manage_universe.py --import-csv ./extra_stocks.csv
```

## Core Empirical Multi-Factor Formula (`v6` Default, `--model v5` Legacy)

Every Friday close (point-in-time, zero look-ahead, Next-Day Open execution, USD FX-normalized in `GLOBAL` mode), winsorized z-scores (`±3.0`) combine 6 peer-reviewed empirical pillars:

`S_v6 = cs_z(0.20 * F_MomQual + 0.15 * F_IndMom + 0.25 * F_NormVal + 0.20 * F_Quality + 0.10 * F_Fund + 0.10 * F_LowRisk)`

1. **`F_MomQual` (20%)**: `0.75 ×` Volatility-Scaled Residual Momentum (Blitz, Huij, & Martens 2011 *JEF*; Barroso & Santa-Clara 2015 *JFE*) `+ 0.25 ×` Frog-in-the-Pan Continuous Information (`FIP`, Da, Gurun, & Warachka 2014 *RFS*).
2. **`F_IndMom` (15%)**: Full-Group 12-week Industry Momentum (Moskowitz & Grinblatt 1999 *JF*; Ehsani & Linnainmaa 2022 *JF*).
3. **`F_NormVal` (25%)**: Capital-structure-neutral within-industry value (`1/PBR` for financials; `0.5 × capped E/P + 0.5 × capped EBITDA/EV` for non-financials, Loughran & Wellman 2011 *JFQA*, strictly positive trailing P/E) `+ 0.5 ×` capped dividend yield (`<= 8.0%`, Boudoukh et al. 2007 *JF*).
4. **`F_Quality` (20%, Microeconomics)**: Lerner excess pricing-power markup (`GrossMargin - peer_median`, Lerner 1934 *RES*; De Loecker et al. 2020 *QJE*), within-industry `ROE` & Operating Margin (Novy-Marx 2013 *JFE*), `FCF/EV` (Ball et al. 2016 *JFE*), liquidity (`CurrentRatio`), minus non-financial leverage (`D/E`, Hou et al. 2015 *RFS*; `QMJ`).
5. **`F_Fund` (10%, Growth & Sequoia Rule of 40)**: Within-industry `Rule of 40 = RevYoY + FCF Margin` (`rule40_%`) combined with Revenue + Earnings growth YoY globally (or point-in-time lagged monthly revenue YoY + acceleration in Taiwan via `backtest_v5.py`).
6. **`F_LowRisk` (10%)**: Betting Against Beta (Frazzini & Pedersen 2014 *JFE*) `+` Low Idiosyncratic Volatility (Ang et al. 2006 *JF*).
7. **Macroeconomic Regime Conditioning (`macro_micro.py`, strictly lagged `shift(1)`)**: Dynamically bounds and tilts pillar weights based on Yield Curve (`^TNX - ^IRX`), Credit Spreads (`HYG/IEF`), `^VIX`, USD (`DX-Y.NYB`), and Oil (`CL=F`) across Credit Stress, Rate/Inflation Shock, and Expansion regimes (disable with `--no-macro`).

Read [empirical-research-v5.md](./references/empirical-research-v5.md) for full LaTeX derivations, peer-reviewed citations, audit tables, and survivorship-bias disclosures.

## Workflow Steps

### 1. Setup & Verify Environment
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest skills/snowball/scripts
```

### 2. Rank & Backtest Any Global Market (`global_ranker.py`)
```bash
# Rank US, Europe, Japan, Hong Kong, Taiwan, or cross-region Global equities
.venv/bin/python skills/snowball/scripts/global_ranker.py --market US --backtest
.venv/bin/python skills/snowball/scripts/global_ranker.py --market JP --top 10
.venv/bin/python skills/snowball/scripts/global_ranker.py --market GLOBAL --backtest

# Rank any custom global watchlist
.venv/bin/python skills/snowball/scripts/global_ranker.py --tickers AAPL,MSFT,GOOGL,ASML.AS,7203.T,2330.TW --benchmark ACWI

# Offline deterministic demo
.venv/bin/python skills/snowball/scripts/global_ranker.py --market GLOBAL --backtest --offline-demo
```

### 3. Deep Taiwan Point-in-Time Audit & Cash-Flow Simulator (`backtest_v5.py`)
Includes FinMind monthly revenue (lagged to the 11th), 5-year rolling median cyclical normalization, and the zero-realized-loss retail odd-lot simulator:
```bash
.venv/bin/python skills/snowball/scripts/backtest_v5.py
```

### 4. Monitor Custom Multi-Market Portfolio & Generate Light/Dark HTML Workbench
1. Copy [portfolio-template.json](./references/portfolio-template.json) to a gitignored local file (`./my_portfolio.json`).
2. Run CLI monitor or generate interactive HTML dashboard:
```bash
.venv/bin/python skills/snowball/scripts/monitor_portfolio.py --portfolio ./my_portfolio.json
.venv/bin/python skills/snowball/scripts/generate_dashboard.py --portfolio ./my_portfolio.json --output ./snowball_dashboard.html
```
3. Apply the **Buy-Only Cash-Flow Rebalancing Rules**:
   - **Core (55%–60%)**: Regular DCA into broad/global/high-dividend ETFs (`VT`, `VOO`, `0056.TW`, etc.).
   - **Satellite (40%–45%)**: Deploy into the Top-20% v5 candidate with highest `Priority = S_v5 + (w_target - w_now) / w_target`.
   - **Freeze Laggards / Trim Overweight Winners Only**: Do not add to stocks falling outside Top 40%; trim profitable holdings (`Price > AvgCost`) only when weight exceeds `2 × w_target` (`20%`) or reaches a user-defined profit target.

### 5. Score Probabilistic Forecasts (Brier Score & BSS)
```bash
.venv/bin/python skills/snowball/scripts/score_predictions.py --csv ./my_predictions.csv
```
