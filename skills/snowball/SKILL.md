---
name: snowball
description: >-
  All-agents global equity multi-factor ranking (v5) and buy-only cash-flow rebalancing skill.
  Covers Global markets (US, Europe, Japan, Hong Kong, Taiwan, or arbitrary custom tickers via Yahoo Finance).
  Use when the user asks to rank global or regional equities using empirical factors (Residual Momentum,
  Full-Group Industry Momentum, Revenue Growth, and 8%-Capped Industry-Neutral Value), run point-in-time
  4-cohort staggered backtests with weekly mark-to-market MDD, score probabilistic forecasts (Brier Score / BSS),
  or generate a Light/Dark interactive HTML portfolio workbench with buy-only cash-flow rebalancing rules.
---

# Snowball — Global Equity Multi-Factor & Cash-Flow Rebalancing Skill (v5)

An agent-agnostic quantitative skill for **global equities** (`US`, `EU`, `JP`, `HK`, `TW`, `GLOBAL`, or custom tickers), compatible with **Antigravity (AGY)**, **Claude Code**, **OpenAI Codex CLI**, **Cursor**, **Windsurf**, **Cline**, and **OpenCode**.

> **Disclaimer**: Strictly for quantitative research, backtesting methodology, and systematic tracking. Not licensed financial advice.

## Supported Global Markets (`global_universe.py`)

| Market Flag | Coverage | Benchmark | Round-Trip Cost Model |
|---|---|---|---|
| `--market US` | 40 liquid US leaders across 10 sectors | `SPY` | `0.10%` |
| `--market EU` | 18 European leaders (Euronext / XETRA / LSE / SIX) | `^STOXX50E` | `0.30%` |
| `--market JP` | 16 Japan leaders (TSE `.T`) | `^N225` | `0.20%` |
| `--market HK` | 12 Hong Kong leaders (HKEX `.HK`) | `^HSI` | `0.45%` (incl. stamp duty) |
| `--market TW` | 84 Taiwan stocks across 20 peer groups (`.TW`/`.TWO`) | `^TWII` | `0.585%` (+ FinMind monthly rev in `backtest_v5.py`) |
| `--market GLOBAL` | Cross-region flagship leaders across US/EU/JP/HK/TW | `ACWI` | `0.30%` |
| `--tickers ...` | Any custom comma-separated Yahoo Finance symbols | Custom (`ACWI` default) | `0.20%` |

## Core v5 Ranking Formula

Every Friday close (point-in-time, zero look-ahead), cross-sectional z-scores (clipped to `±3.0`) are combined:

`S_v5 = cs_z(0.5 * F_ResMom_26_4w + 0.5 * F_IndMom_Full_12w + 1.0 * F_Fund + 1.5 * F_NormVal)`

1. **`F_ResMom_26_4w`** (Residual Momentum, Blitz et al. 2011): 26-week skipping last 4-week cumulative Beta-adjusted alpha.
2. **`F_IndMom_Full_12w`** (Full-Group Industry Momentum, Moskowitz & Grinblatt 1999): 12-week peer-group mean alpha (full-group inclusive to prevent small-group leave-one-out reversal bias).
3. **`F_Fund`** (Fundamental Growth): Revenue growth YoY from Yahoo Finance globally (or point-in-time lagged monthly revenue YoY + acceleration in Taiwan via `backtest_v5.py`).
4. **`F_NormVal`** (8%-Capped Industry-Neutral Value): Within-sector `1/PBR` (financials) or capped `1/PER` (`E/P <= 15%`) + `0.5 ×` capped dividend yield (`<= 8.0%`).

Read [empirical-research-v5.md](./references/empirical-research-v5.md) for academic citations, audit tables, and survivorship-bias disclosures.

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
