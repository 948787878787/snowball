# Snowball 

An open-source **Antigravity Agent Skill** & Python quantitative research workbench for **Taiwan equity multi-factor ranking (v5)** and **buy-only retail cash-flow rebalancing** — built to compound high-cash-flow equity snowballs systematically.

> **Disclaimer**: This repository is provided strictly for quantitative research, backtesting methodology, and educational purposes. It does not constitute financial or investment advice.

## What's Inside

- **`skills/tw-quant-factor-v5/SKILL.md`**: Ready-to-use Antigravity Skill definition with progressive disclosure.
- **`skills/tw-quant-factor-v5/references/empirical-research-v5.md`**: Full academic literature review, v1 → v5 formula audit, and honest out-of-sample performance summary.
- **`skills/tw-quant-factor-v5/references/portfolio-template.json`**: Zero-PII JSON template for plugging in your own local portfolio.
- **`skills/tw-quant-factor-v5/scripts/`**:
  - `backtest_v5.py`: Point-in-time v5 multi-factor engine (Residual Momentum + Full-Group Industry Momentum + Monthly Revenue Momentum + 8%-Capped Cyclical-Normalized Industry-Neutral Value), 4-cohort staggered evaluation, true weekly mark-to-market MDD, and retail odd-lot cash-flow simulator.
  - `monitor_portfolio.py`: Generic CLI monitor that reads a local JSON portfolio file and outputs v5 rankings, cash-flow rebalancing priorities, and overweight winner-trim alerts.
  - `generate_dashboard.py`: Generates a self-contained Light/Dark interactive HTML workbench (`resources/dashboard_template.html`) with `localStorage` persistence, editable holdings, a monthly Core+Satellite odd-lot order calculator, and a 10-year compounding simulator.
  - `score_predictions.py`: Probabilistic forecast tracker and Brier Score evaluator.
  - `tests/`: 29 pytest unit tests validating factor math, caps, causality, and zero-realized-loss invariants.

## Installation

### 1. Install as an Antigravity Skill
Copy `skills/tw-quant-factor-v5` into either your project's `.agents/skills/` directory or your global `~/.gemini/config/skills/` directory:

```bash
mkdir -p ~/.gemini/config/skills
cp -r skills/tw-quant-factor-v5 ~/.gemini/config/skills/
```

### 2. Set Up Python Environment & Run Tests
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest skills/tw-quant-factor-v5/scripts
```

### 3. Run the v5 Backtest or Portfolio Monitor
```bash
# Run full v5 point-in-time backtest (downloads & caches Yahoo Finance + FinMind data locally)
.venv/bin/python skills/tw-quant-factor-v5/scripts/backtest_v5.py

# Copy template to a local gitignored file and run weekly monitor
cp skills/tw-quant-factor-v5/references/portfolio-template.json ./my_portfolio.json
.venv/bin/python skills/tw-quant-factor-v5/scripts/monitor_portfolio.py --portfolio ./my_portfolio.json

# Generate interactive Light/Dark HTML dashboard
.venv/bin/python skills/tw-quant-factor-v5/scripts/generate_dashboard.py --portfolio ./my_portfolio.json --output ./v5_dashboard.html
```

## License

MIT License. See [LICENSE](./LICENSE).
