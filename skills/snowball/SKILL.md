---
name: snowball
description: >-
  All-agents Taiwan equity multi-factor ranking (v5) and buy-only cash-flow rebalancing skill.
  Use when the user asks to rank Taiwan equities using empirical factors (Residual Momentum,
  Full-Group Industry Momentum, Monthly Revenue Momentum, and 8%-Capped Industry-Neutral Value),
  run point-in-time backtests with 4 staggered cohorts and weekly mark-to-market MDD,
  score probabilistic forecasts (Brier Score / BSS), or generate a Light/Dark interactive HTML
  portfolio workbench with buy-only cash-flow rebalancing and overweight winner-trim rules.
---

# Snowball — Taiwan Equity Multi-Factor & Cash-Flow Rebalancing Skill (v5)

An agent-agnostic quantitative skill compatible with **Antigravity (AGY)**, **Claude Code**, **OpenAI Codex CLI**, **Cursor**, **Windsurf**, **Cline**, and **OpenCode**.

> **Disclaimer**: Strictly for quantitative research, backtesting education, and systematic portfolio tracking. Not licensed financial advice.

## Core v5 Ranking Formula

Every Friday close (point-in-time, zero look-ahead), cross-sectional z-scores (clipped to `±3.0`) are computed across the peer-group universe:

- **v5 Core (3-Pillar)**: `cs_z(F_ResMom_26_4w + F_FinStmt + F_NormVal)`
- **v5 High-Income Defensive (Recommended Satellite)**:
  `cs_z(0.5 * F_ResMom_26_4w + 0.5 * F_IndMom_Full_12w + F_FinStmt + 1.5 * F_NormVal)`

Factors:
1. **`F_ResMom_26_4w`** (Residual Momentum, Blitz et al. 2011): 26-week skipping last 4-week cumulative Beta-adjusted alpha.
2. **`F_IndMom_Full_12w`** (Full-Group Industry Momentum, Moskowitz & Grinblatt 1999): 12-week peer-group mean alpha (full-group inclusive to avoid `-1/(G-1)` small-group reversal artifacts).
3. **`F_FinStmt`** (Point-in-time Revenue Momentum): 50% trailing 3-month revenue YoY + 50% 3-month YoY acceleration, lagged to the 11th of the following month.
4. **`F_NormVal`** (8%-Capped Industry-Neutral Value): Within-peer-group standardized value (`1/PBR` for financials, normalized `1/PER` for non-financials) + `0.5 ×` cross-sectional capped dividend yield (`Yield <= 8.0%`, `E/P <= 15%`, 5-year rolling median for cyclical sectors).

Read [empirical-research-v5.md](./references/empirical-research-v5.md) for academic citations, audit tables, and survivorship-bias disclosures.

## Workflow Steps

### 1. Setup & Verify Environment
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest skills/snowball/scripts
```

### 2. Run Point-in-Time Backtest & Audit (v5)
Runs 4-cohort staggered Jegadeesh-Titman evaluation, true weekly mark-to-market MDD, Newey-West (`lags=6`) t-stats, Jensen's Alpha, and the retail odd-lot cash-flow simulator:
```bash
.venv/bin/python skills/snowball/scripts/backtest_v5.py
```

### 3. Monitor Any Custom Portfolio (Zero-PII)
1. Copy [portfolio-template.json](./references/portfolio-template.json) to a local gitignored path (e.g., `./my_portfolio.json`).
2. Run the CLI monitor:
```bash
.venv/bin/python skills/snowball/scripts/monitor_portfolio.py --portfolio ./my_portfolio.json
```
3. Apply the **Buy-Only Cash-Flow Rebalancing Rules**:
   - **Core (55%–60% monthly cash flow)**: Fixed DCA into broad/high-dividend ETFs.
   - **Satellite (40%–45% monthly cash flow)**: Deploy into the Top-20% v5 candidate with highest `Priority = S_v5 + (w_target - w_now) / w_target`.
   - **Freeze Laggards / Trim Overweight Winners Only**: Do not add to stocks falling outside Top 40%; trim profitable holdings (`Price > AvgCost`) only when weight exceeds `2 × w_target` (`20%`) or reaches a user-defined profit target.

### 4. Generate Interactive Light/Dark HTML Workbench
```bash
# Live mode (uses cached or fetched market data; falls back cleanly if offline)
.venv/bin/python skills/snowball/scripts/generate_dashboard.py --portfolio ./my_portfolio.json --output ./snowball_dashboard.html

# Instant offline demo mode
.venv/bin/python skills/snowball/scripts/generate_dashboard.py --offline --output ./snowball_dashboard.html
```

### 5. Score Probabilistic Forecasts (Brier Score & BSS)
```bash
.venv/bin/python skills/snowball/scripts/score_predictions.py --csv ./my_predictions.csv
```
