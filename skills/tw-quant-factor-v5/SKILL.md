---
name: tw-quant-factor-v5
description: >-
  Taiwan equity multi-factor ranking (v5) and buy-only cash-flow rebalancing skill.
  Use when the user asks to rank Taiwan stocks using empirical factors (Residual Momentum,
  Full-Group Industry Momentum, Monthly Revenue Momentum, and 8%-Capped Industry-Neutral Value),
  run point-in-time backtests with 4 staggered cohorts and weekly mark-to-market MDD,
  or generate weekly buy-only / winner-trim rebalancing guidance for a retail portfolio.
---

# Taiwan Equity Multi-Factor & Cash-Flow Rebalancing (v5)

This skill implements an empirically audited, point-in-time multi-factor ranking engine (**v5**) for Taiwan listed equities, paired with a **Core + Satellite, Buy-Only Cash-Flow Rebalancing** workflow designed for retail odd-lot investors who prefer low turnover and zero realized losses.

> **Disclaimer**: This skill is for quantitative research, educational analysis, and systematic tracking only. It is not licensed financial advice. All investment decisions and position sizing remain the user's responsibility.

## Core Methodology (v5 Formula)

Every Friday close (point-in-time, zero look-ahead), cross-sectional z-scores (clipped at `±3.0`) are computed across the peer-group universe:

- **v5 Core (3-Pillar)**: `cs_z(F_ResMom_26_4w + F_FinStmt + F_NormVal)`
- **v5 High-Income Defensive (Recommended Satellite Engine)**:
  `cs_z(0.5 * F_ResMom_26_4w + 0.5 * F_IndMom_Full_12w + F_FinStmt + 1.5 * F_NormVal)`

Where:
1. **`F_ResMom_26_4w`** (Residual Momentum, Blitz et al. 2011): 26-week skipping last 4-week cumulative Beta-adjusted alpha.
2. **`F_IndMom_Full_12w`** (Full-Group Industry Momentum, Moskowitz & Grinblatt 1999): 12-week peer-group mean alpha (full-group inclusive to avoid `-1/(G-1)` leave-one-out reversal artifacts in small groups).
3. **`F_FinStmt`** (Point-in-time Revenue Momentum): 50% trailing 3-month revenue YoY + 50% 3-month YoY acceleration, strictly lagged to the 11th of the following month.
4. **`F_NormVal`** (Cyclical-Normalized & Capped Industry-Neutral Value): Within-peer-group standardized value (`1/PBR` for financials, normalized `1/PER` for non-financials) + `0.5 ×` cross-sectional capped dividend yield (`Yield <= 8.0%`, `E/P <= 15%`, 5-year rolling median for cyclical sectors).

For full academic citations, v1→v5 evolution, and honest out-of-sample audit results (including survivorship-bias disclosures), read [empirical-research-v5.md](./references/empirical-research-v5.md).

## Workflow Steps

### 1. Environment Setup & Verification
Install dependencies in a virtual environment and run the unit test suite (`29` tests):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest skills/tw-quant-factor-v5/scripts
```

### 2. Run Full Point-in-Time Backtest & Audit (v5)
Executes 4-cohort staggered Jegadeesh-Titman evaluation, true weekly mark-to-market MDD, Newey-West (`lags=6`) t-stats, Jensen's Alpha vs. equal-weight total return, and the retail odd-lot cash-flow simulator:

```bash
.venv/bin/python skills/tw-quant-factor-v5/scripts/backtest_v5.py
```

### 3. Monitor a User's Custom Portfolio (Zero-PII Template)
1. Ask the user to copy [portfolio-template.json](./references/portfolio-template.json) to a **local, gitignored** path (e.g., `./my_portfolio.json`) and fill in their own holdings, cost basis, target sell prices, and watchlist.
2. Run the generic portfolio monitor:

```bash
.venv/bin/python skills/tw-quant-factor-v5/scripts/monitor_portfolio.py --portfolio ./my_portfolio.json
```

3. Apply the **Buy-Only Cash-Flow Rebalancing Rules**:
   - **Core (55%–60% of monthly inflow)**: Regular fixed-amount DCA into broad market or high-dividend ETFs.
   - **Satellite (40%–45% of monthly inflow)**: Deploy into the Top-20% v5 candidate (price suitable for odd lots) with the highest cash-flow priority `Priority = S_v5 + (w_target - w_now) / w_target`.
   - **Freeze Losers / Trim Overweight Winners Only**: Do not add to stocks falling outside Top 40%; trim profitable holdings (`Price > AvgCost`) only when weight exceeds `2 × w_target` (`20%`) or reaches a user-defined profit target.

### 4. Score Weekly Probabilistic Forecasts
If tracking directional predictions in a CSV log, run [score_predictions.py](./scripts/score_predictions.py) to compute hit rates and Brier Scores (`BS = mean((p - o)^2)`).
